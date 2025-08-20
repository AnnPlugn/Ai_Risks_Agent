# src/agents/profiler_agent.py
"""
Улучшенный профайлер-агент для сбора и анализа данных об ИИ-агенте
Новая архитектура с контекстно-осознанным чанкингом и LangGraph оркестрацией
"""
import os
import json
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from .base_agent import AnalysisAgent, AgentConfig
from ..models.risk_models import AgentProfile, AgentTaskResult, ProcessingStatus, AgentType, AutonomyLevel, \
    DataSensitivity, WorkflowState
from ..tools.document_parser import create_document_parser, parse_agent_documents
from ..tools.code_analyzer import create_code_analyzer, analyze_agent_codebase
from ..tools.prompt_analyzer import create_prompt_analyzer, analyze_agent_prompts
from ..utils.logger import LogContext, get_logger
from ..prompts.profiler_prompts import profiler_system_prompt, json_profiler_extr_prompt, base_prompts_profiler, summary_report_prompt


class FileType(Enum):
    """Типы файлов для анализа"""
    CODE = "code"
    DOCUMENT = "document"
    CONFIG = "config"
    PROMPT = "prompt"
    UNKNOWN = "unknown"


@dataclass
class FileMetadata:
    """Метаданные файла с приоритетом"""
    path: str
    file_type: FileType
    size_bytes: int
    last_modified: datetime
    encoding: Optional[str] = None
    language: Optional[str] = None
    priority: int = 0
    content_hash: Optional[str] = None


@dataclass
class ContextChunk:
    """Контекстный чанк для обработки LLM"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    context_type: str
    priority: int
    size_tokens: int
    related_chunks: List[str] = None
    processing_hints: List[str] = None


@dataclass
class ProcessingStage:
    """Стадия обработки с метриками"""
    stage_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "in_progress"
    output_files: List[str] = None
    metrics: Dict[str, Any] = None
    error_details: Optional[str] = None


class FileSystemCrawler:
    """Улучшенный компонент для сканирования файловой системы"""

    def __init__(self, logger):
        self.logger = logger
        self.supported_extensions = {
            # Код
            '.py': FileType.CODE, '.js': FileType.CODE, '.jsx': FileType.CODE,
            '.ts': FileType.CODE, '.tsx': FileType.CODE, '.java': FileType.CODE,
            '.cpp': FileType.CODE, '.c': FileType.CODE, '.cs': FileType.CODE,
            '.go': FileType.CODE, '.rs': FileType.CODE, '.php': FileType.CODE,
            '.rb': FileType.CODE,

            # Документы
            '.docx': FileType.DOCUMENT, '.xlsx': FileType.DOCUMENT,
            '.pdf': FileType.DOCUMENT, '.md': FileType.DOCUMENT,
            '.txt': FileType.DOCUMENT, '.rtf': FileType.DOCUMENT,

            # Конфигурации
            '.json': FileType.CONFIG, '.yaml': FileType.CONFIG, '.yml': FileType.CONFIG,
            '.toml': FileType.CONFIG, '.ini': FileType.CONFIG, '.env': FileType.CONFIG,
            '.xml': FileType.CONFIG,

            # Промпты
            '.prompt': FileType.PROMPT, '.instruction': FileType.PROMPT
        }

        self.exclude_patterns = {
            'node_modules', '__pycache__', '.git', '.venv', 'venv',
            'target', 'build', 'dist', '.idea', '.vscode',
            'logs', 'tmp', 'temp', '.pytest_cache', '.mypy_cache'
        }

        self.cache = {}  # Кэш для повторных сканирований

    async def crawl_sources(self, source_paths: List[str], max_files: int = 200) -> List[FileMetadata]:
        """Оптимизированное сканирование с кэшированием"""
        cache_key = self._generate_cache_key(source_paths, max_files)

        if cache_key in self.cache:
            self.logger.bind_context("profiler", "crawler").info("Используем кэшированные результаты")
            return self.cache[cache_key]

        all_files = []
        processed_paths = set()

        for source_path in source_paths:
            path = Path(source_path)
            abs_path = path.absolute()

            if str(abs_path) in processed_paths:
                continue
            processed_paths.add(str(abs_path))

            if path.is_file():
                file_meta = await self._analyze_single_file(path)
                if file_meta:
                    all_files.append(file_meta)
            elif path.is_dir():
                dir_files = await self._crawl_directory_optimized(path, max_files - len(all_files))
                all_files.extend(dir_files)

                if len(all_files) >= max_files:
                    break

        # Приоритизация и ограничение
        prioritized_files = self._prioritize_files_advanced(all_files)

        if len(prioritized_files) > max_files:
            self.logger.bind_context("profiler", "crawler").warning(
                f"Ограничиваем {len(prioritized_files)} файлов до {max_files}"
            )
            prioritized_files = prioritized_files[:max_files]

        # Кэшируем результат
        self.cache[cache_key] = prioritized_files

        return prioritized_files

    def _generate_cache_key(self, source_paths: List[str], max_files: int) -> str:
        """Генерация ключа кэша"""
        paths_str = "|".join(sorted([str(Path(p).absolute()) for p in source_paths]))
        return hashlib.md5(f"{paths_str}_{max_files}".encode()).hexdigest()

    async def _crawl_directory_optimized(self, directory: Path, remaining_slots: int) -> List[FileMetadata]:
        """Оптимизированное сканирование директории"""
        if remaining_slots <= 0:
            return []

        files = []
        try:
            # Используем iterdir() вместо rglob() для лучшей производительности
            files_found = 0

            for item in directory.rglob('*'):
                if files_found >= remaining_slots:
                    break

                if (item.is_file() and
                        self._should_include_file(item) and
                        not any(exclude in str(item) for exclude in self.exclude_patterns)):

                    file_meta = await self._analyze_single_file(item)
                    if file_meta:
                        files.append(file_meta)
                        files_found += 1

        except (PermissionError, OSError) as e:
            self.logger.bind_context("profiler", "crawler").warning(
                f"Ошибка доступа к {directory}: {e}"
            )

        return files

    def _prioritize_files_advanced(self, files: List[FileMetadata]) -> List[FileMetadata]:
        """Продвинутая приоритизация файлов"""
        for file_meta in files:
            file_meta.priority = self._calculate_advanced_priority(file_meta)

        # Сортировка: сначала по приоритету, потом по размеру
        return sorted(files, key=lambda f: (-f.priority, f.size_bytes), reverse=False)

    def _calculate_advanced_priority(self, file_meta: FileMetadata) -> int:
        """Улучшенный расчет приоритета"""
        priority = 0
        file_name = Path(file_meta.path).name.lower()

        # Базовые приоритеты по типу
        type_priorities = {
            FileType.PROMPT: 15,
            FileType.CONFIG: 12,
            FileType.DOCUMENT: 8,
            FileType.CODE: 5,
            FileType.UNKNOWN: 1
        }
        priority += type_priorities.get(file_meta.file_type, 1)

        # Бонусы за важные файлы
        important_keywords = {
            'readme': 5, 'main': 5, 'index': 5, 'app': 5,
            'config': 4, 'settings': 4, 'prompt': 8, 'system': 8,
            'guardrail': 8, 'instruction': 7, 'agent': 6, 'bot': 6,
            'api': 3, 'model': 4, 'train': 3, 'test': 2
        }

        for keyword, bonus in important_keywords.items():
            if keyword in file_name:
                priority += bonus

        # Штраф за слишком большие файлы
        if file_meta.size_bytes > 1024 * 1024:  # > 1MB
            priority -= 3
        elif file_meta.size_bytes > 100 * 1024:  # > 100KB
            priority -= 1

        # Бонус за недавние изменения
        if hasattr(file_meta, 'last_modified'):
            days_old = (datetime.now() - file_meta.last_modified).days
            if days_old < 7:
                priority += 2
            elif days_old < 30:
                priority += 1

        return max(priority, 1)

    async def _analyze_single_file(self, file_path: Path) -> Optional[FileMetadata]:
        """Анализ файла с хэшированием для дедупликации"""
        try:
            stat = file_path.stat()

            # Генерируем хэш содержимого для дедупликации
            content_hash = None
            if stat.st_size < 10 * 1024 * 1024:  # Только для файлов < 10MB
                try:
                    with open(file_path, 'rb') as f:
                        content_hash = hashlib.md5(f.read()).hexdigest()
                except Exception:
                    pass

            return FileMetadata(
                path=str(file_path),
                file_type=self._determine_file_type(file_path),
                size_bytes=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                encoding=await self._detect_encoding(file_path),
                language=self._detect_language(file_path),
                content_hash=content_hash
            )

        except Exception as e:
            self.logger.bind_context("profiler", "crawler").error(f"Ошибка анализа {file_path}: {e}")
            return None

    def _determine_file_type(self, file_path: Path) -> FileType:
        """Улучшенное определение типа файла"""
        extension = file_path.suffix.lower()
        base_type = self.supported_extensions.get(extension, FileType.UNKNOWN)

        # Специальная логика для текстовых файлов
        if base_type == FileType.DOCUMENT and extension in ['.txt', '.md']:
            file_name_lower = file_path.name.lower()
            prompt_keywords = ['prompt', 'instruction', 'system', 'guardrail', 'rule']
            if any(keyword in file_name_lower for keyword in prompt_keywords):
                return FileType.PROMPT

        return base_type

    def _detect_language(self, file_path: Path) -> Optional[str]:
        """Определение языка программирования"""
        lang_map = {
            '.py': 'python', '.js': 'javascript', '.jsx': 'javascript',
            '.ts': 'typescript', '.tsx': 'typescript', '.java': 'java',
            '.cpp': 'cpp', '.c': 'c', '.cs': 'csharp', '.go': 'go',
            '.rs': 'rust', '.php': 'php', '.rb': 'ruby'
        }
        return lang_map.get(file_path.suffix.lower())

    async def _detect_encoding(self, file_path: Path) -> Optional[str]:
        """Быстрое определение кодировки"""
        if file_path.suffix.lower() in ['.pdf', '.docx', '.xlsx']:
            return None

        encodings = ['utf-8', 'cp1251', 'latin-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(512)  # Читаем только начало
                return encoding
            except UnicodeDecodeError:
                continue
            except Exception:
                break
        return 'utf-8'

    def _should_include_file(self, file_path: Path) -> bool:
        """Проверка включения файла с дополнительными фильтрами"""
        # Размер файла
        try:
            size = file_path.stat().st_size
            if size > 50 * 1024 * 1024:  # > 50MB
                return False
            if size == 0:  # Пустые файлы
                return False
        except Exception:
            return False

        # Расширение
        if file_path.suffix.lower() not in self.supported_extensions:
            return False

        # Временные файлы
        name = file_path.name.lower()
        if name.startswith('.') or name.endswith('.tmp') or name.endswith('.bak'):
            return False

        return True


class ContextAwareChunker:
    """Продвинутый контекстно-осознанный чанкер"""

    def __init__(self, max_chunk_size: int = 6000, overlap_size: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.context_analyzers = {
            'agent_overview': self._analyze_overview_context,
            'technical_architecture': self._analyze_technical_context,
            'prompts_and_instructions': self._analyze_prompt_context,
            'business_logic': self._analyze_business_context,
            'configurations': self._analyze_config_context,
            'supporting_docs': self._analyze_support_context
        }

    async def create_chunks(self, files_metadata: List[FileMetadata], parsed_data: Dict[str, Any]) -> List[
        ContextChunk]:
        """Создание оптимизированных контекстных чанков"""
        # Дедупликация по хэшам
        unique_files = self._deduplicate_files(files_metadata)

        # Группировка по контексту
        context_groups = self._group_by_context_smart(unique_files, parsed_data)

        chunks = []
        for context_type, context_data in context_groups.items():
            context_chunks = await self._create_context_chunks(context_type, context_data)
            chunks.extend(context_chunks)

        return self._optimize_chunk_distribution(chunks)

    def _deduplicate_files(self, files_metadata: List[FileMetadata]) -> List[FileMetadata]:
        """Дедупликация файлов по хэшу содержимого"""
        seen_hashes = set()
        unique_files = []

        for file_meta in files_metadata:
            if file_meta.content_hash:
                if file_meta.content_hash not in seen_hashes:
                    seen_hashes.add(file_meta.content_hash)
                    unique_files.append(file_meta)
            else:
                unique_files.append(file_meta)

        return unique_files

    def _group_by_context_smart(self, files_metadata: List[FileMetadata], parsed_data: Dict[str, Any]) -> Dict[
        str, Dict[str, Any]]:
        """Интеллектуальная группировка по контексту"""
        context_groups = {
            'agent_overview': {'priority': 10, 'data': {}, 'analyzer': 'overview'},
            'technical_architecture': {'priority': 9, 'data': {}, 'analyzer': 'technical'},
            'prompts_and_instructions': {'priority': 8, 'data': {}, 'analyzer': 'prompts'},
            'business_logic': {'priority': 7, 'data': {}, 'analyzer': 'business'},
            'configurations': {'priority': 6, 'data': {}, 'analyzer': 'config'},
            'supporting_docs': {'priority': 5, 'data': {}, 'analyzer': 'support'}
        }

        # Автоматическое распределение файлов
        for file_meta in files_metadata:
            context_type = self._classify_file_context(file_meta)
            file_path = file_meta.path

            # Получаем данные файла
            file_data = self._extract_file_data(file_path, parsed_data)
            if file_data:
                context_groups[context_type]['data'][file_path] = file_data

        # Добавляем агрегированные данные
        if 'code_analysis' in parsed_data:
            context_groups['technical_architecture']['data']['_code_summary'] = parsed_data['code_analysis']

        if 'prompt_analysis' in parsed_data:
            context_groups['prompts_and_instructions']['data']['_prompt_summary'] = parsed_data['prompt_analysis']

        return context_groups

    def _classify_file_context(self, file_meta: FileMetadata) -> str:
        """Классификация файла по контексту"""
        path = Path(file_meta.path)
        name_lower = path.name.lower()

        # Приоритетная классификация
        if file_meta.file_type == FileType.PROMPT:
            return 'prompts_and_instructions'

        if any(keyword in name_lower for keyword in ['readme', 'overview', 'introduction']):
            return 'agent_overview'

        if any(keyword in name_lower for keyword in ['architecture', 'technical', 'design']):
            return 'technical_architecture'

        if any(keyword in name_lower for keyword in ['config', 'setting', 'env']):
            return 'configurations'

        if any(keyword in name_lower for keyword in ['prompt', 'instruction', 'system']):
            return 'prompts_and_instructions'

        if file_meta.file_type == FileType.CODE:
            if any(keyword in name_lower for keyword in ['main', 'app', 'index', 'agent', 'bot']):
                return 'technical_architecture'
            else:
                return 'business_logic'

        return 'supporting_docs'

    async def _create_context_chunks(self, context_type: str, context_data: Dict[str, Any]) -> List[ContextChunk]:
        """Создание чанков для контекста с интеллектуальным разделением"""
        if not context_data['data']:
            return []

        # Получаем функцию анализатора, используя 'default' как запасной вариант
        analyzer_func = self.context_analyzers.get(context_data.get('analyzer', 'default'))

        # Анализируем контекст для определения стратегии чанкинга
        analysis_result = {}
        if analyzer_func and 'data' in context_data:
            try:
                analysis_result = analyzer_func(context_data['data'])
            except Exception as e:
                # Можно добавить логирование ошибки
                print(f"Ошибка при анализе контекста: {e}")

        # Собираем контент
        content_sections = []
        metadata = {
            'context_type': context_type,
            'priority': context_data['priority'],
            'files_count': len(context_data['data']),
            'analysis': analysis_result
        }

        for file_path, file_data in context_data['data'].items():
            section_content = self._format_file_content(file_path, file_data)
            if section_content:
                content_sections.append(section_content)

        if not content_sections:
            return []

        # Создаем чанки в зависимости от размера
        full_content = "\n\n".join(content_sections)

        if len(full_content) <= self.max_chunk_size:
            # Один чанк
            return [ContextChunk(
                chunk_id=f"{context_type}_complete",
                content=full_content,
                metadata=metadata,
                context_type=context_type,
                priority=context_data['priority'],
                size_tokens=len(full_content) // 4,
                processing_hints=analysis_result.get('processing_hints', [])
            )]
        else:
            # Множественные чанки
            return self._split_content_intelligently(context_type, content_sections, metadata)

    def _format_file_content(self, file_path: str, file_data: Any) -> str:
        """Форматирование содержимого файла"""
        file_name = Path(file_path).name

        if isinstance(file_data, dict):
            if 'content' in file_data:
                return f"=== {file_name} ===\n{file_data['content']}"
            elif 'sections' in file_data:
                sections = []
                for section_name, section_content in file_data['sections'].items():
                    sections.append(f"## {section_name}\n{section_content}")
                return f"=== {file_name} ===\n" + "\n\n".join(sections)
            else:
                return f"=== {file_name} ===\n{json.dumps(file_data, ensure_ascii=False, indent=2)}"
        else:
            return f"=== {file_name} ===\n{str(file_data)}"

    def _split_content_intelligently(self, context_type: str, content_sections: List[str], metadata: Dict[str, Any]) -> \
    List[ContextChunk]:
        """Интеллектуальное разделение контента"""
        chunks = []
        current_chunk_content = []
        current_size = 0
        chunk_counter = 1

        for section in content_sections:
            section_size = len(section)

            if current_size + section_size <= self.max_chunk_size:
                current_chunk_content.append(section)
                current_size += section_size
            else:
                # Создаем чанк из накопленного контента
                if current_chunk_content:
                    chunks.append(self._create_chunk_from_content(
                        context_type, current_chunk_content, metadata, chunk_counter
                    ))
                    chunk_counter += 1

                # Начинаем новый чанк
                if section_size <= self.max_chunk_size:
                    current_chunk_content = [section]
                    current_size = section_size
                else:
                    # Секция слишком большая - разбиваем её
                    sub_chunks = self._split_large_section(section, self.max_chunk_size)
                    for sub_chunk in sub_chunks:
                        chunks.append(self._create_chunk_from_content(
                            context_type, [sub_chunk], metadata, chunk_counter
                        ))
                        chunk_counter += 1
                    current_chunk_content = []
                    current_size = 0

        # Последний чанк
        if current_chunk_content:
            chunks.append(self._create_chunk_from_content(
                context_type, current_chunk_content, metadata, chunk_counter
            ))

        # Устанавливаем связи между чанками
        self._establish_chunk_relationships(chunks)

        return chunks

    def _create_chunk_from_content(self, context_type: str, content_list: List[str], metadata: Dict[str, Any],
                                   chunk_num: int) -> ContextChunk:
        """Создание чанка из контента"""
        content = "\n\n".join(content_list)
        chunk_metadata = metadata.copy()
        chunk_metadata.update({
            'chunk_number': chunk_num,
            'content_sections': len(content_list)
        })

        return ContextChunk(
            chunk_id=f"{context_type}_part_{chunk_num}",
            content=content,
            metadata=chunk_metadata,
            context_type=context_type,
            priority=metadata['priority'],
            size_tokens=len(content) // 4,
            processing_hints=metadata.get('analysis', {}).get('processing_hints', [])
        )

    def _establish_chunk_relationships(self, chunks: List[ContextChunk]):
        """Установка связей между чанками"""
        if len(chunks) <= 1:
            return

        for i, chunk in enumerate(chunks):
            related = []
            if i > 0:
                related.append(chunks[i - 1].chunk_id)
            if i < len(chunks) - 1:
                related.append(chunks[i + 1].chunk_id)
            chunk.related_chunks = related

    # Анализаторы контекста
    def _analyze_overview_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ контекста обзора агента"""
        return {
            'focus_areas': ['agent_name', 'primary_purpose', 'target_audience'],
            'processing_hints': ['extract_high_level_description', 'identify_main_capabilities']
        }

    def _analyze_technical_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ технического контекста"""
        return {
            'focus_areas': ['architecture', 'dependencies', 'frameworks', 'apis'],
            'processing_hints': ['extract_technical_stack', 'identify_integrations', 'assess_complexity']
        }

    def _analyze_prompt_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ контекста промптов"""
        return {
            'focus_areas': ['system_prompts', 'instructions', 'guardrails', 'personality'],
            'processing_hints': ['extract_behavioral_rules', 'identify_restrictions', 'assess_autonomy']
        }

    def _analyze_business_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ бизнес-контекста"""
        return {
            'focus_areas': ['business_logic', 'workflows', 'decision_making'],
            'processing_hints': ['extract_business_rules', 'identify_decision_points']
        }

    def _analyze_config_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ конфигурационного контекста"""
        return {
            'focus_areas': ['settings', 'parameters', 'environment_variables'],
            'processing_hints': ['extract_configuration_parameters', 'identify_security_settings']
        }

    def _analyze_support_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ вспомогательного контекста"""
        return {
            'focus_areas': ['documentation', 'examples', 'guides'],
            'processing_hints': ['extract_usage_examples', 'identify_best_practices']
        }

    def _extract_file_data(self, file_path: str, parsed_data: Dict[str, Any]) -> Optional[Any]:
        """Извлечение данных файла из parsed_data"""
        if file_path in parsed_data.get('documents', {}):
            return parsed_data['documents'][file_path]
        elif file_path in parsed_data.get('code_files', {}):
            return parsed_data['code_files'][file_path]
        elif file_path in parsed_data.get('config_files', {}):
            return parsed_data['config_files'][file_path]
        elif file_path in parsed_data.get('prompt_files', {}):
            return parsed_data['prompt_files'][file_path]
        return None

    def _split_large_section(self, section: str, max_size: int) -> List[str]:
        """Разбиение большой секции"""
        lines = section.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 для \n

            if current_size + line_size <= max_size:
                current_chunk.append(line)
                current_size += line_size
            else:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def _optimize_chunk_distribution(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Оптимизация распределения чанков"""
        # Сортировка по приоритету
        chunks.sort(key=lambda c: c.priority, reverse=True)

        # Балансировка размеров
        target_size = self.max_chunk_size * 0.8  # 80% от максимального размера

        optimized_chunks = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i]

            # Если чанк слишком маленький, пытаемся объединить со следующим
            if (current_chunk.size_tokens < target_size * 0.3 and  # < 30% от целевого размера
                    i + 1 < len(chunks) and
                    chunks[i + 1].context_type == current_chunk.context_type):

                next_chunk = chunks[i + 1]
                combined_size = current_chunk.size_tokens + next_chunk.size_tokens

                if combined_size <= self.max_chunk_size:
                    # Объединяем чанки
                    combined_chunk = ContextChunk(
                        chunk_id=f"{current_chunk.context_type}_combined_{len(optimized_chunks)}",
                        content=f"{current_chunk.content}\n\n{next_chunk.content}",
                        metadata={**current_chunk.metadata,
                                  'combined_from': [current_chunk.chunk_id, next_chunk.chunk_id]},
                        context_type=current_chunk.context_type,
                        priority=max(current_chunk.priority, next_chunk.priority),
                        size_tokens=combined_size,
                        processing_hints=list(
                            set(current_chunk.processing_hints or []) | set(next_chunk.processing_hints or []))
                    )
                    optimized_chunks.append(combined_chunk)
                    i += 2  # Пропускаем следующий чанк
                else:
                    optimized_chunks.append(current_chunk)
                    i += 1
            else:
                optimized_chunks.append(current_chunk)
                i += 1

        return optimized_chunks


class LLMOrchestrator:
    """Продвинутый оркестратор с LangGraph интеграцией"""

    def __init__(self, agent: 'EnhancedProfilerAgent'):
        self.agent = agent
        self.logger = agent.logger
        self.processing_cache = {}

    async def process_chunks(self, chunks: List[ContextChunk], assessment_id: str) -> Dict[str, Any]:
        """Оркестрация обработки чанков через LLM"""

        # Группировка по контексту для параллельной обработки
        context_groups = self._group_chunks_by_context(chunks)

        # Определение порядка обработки
        processing_order = self._determine_processing_order(context_groups)

        results = {}

        for context_type in processing_order:
            group_chunks = context_groups[context_type]

            self.logger.bind_context(assessment_id, "llm_orchestrator").info(
                f"Обработка контекста: {context_type} ({len(group_chunks)} чанков)"
            )

            try:
                # Параллельная обработка чанков контекста
                group_result = await self._process_context_group_parallel(
                    context_type, group_chunks, assessment_id
                )
                results[context_type] = group_result

                # Короткая пауза между контекстами
                await asyncio.sleep(0.5)

            except Exception as e:
                self.logger.bind_context(assessment_id, "llm_orchestrator").error(
                    f"Ошибка обработки контекста {context_type}: {e}"
                )
                results[context_type] = {"error": str(e), "context_type": context_type}

        return results

    def _group_chunks_by_context(self, chunks: List[ContextChunk]) -> Dict[str, List[ContextChunk]]:
        """Группировка чанков по контексту"""
        groups = {}
        for chunk in chunks:
            if chunk.context_type not in groups:
                groups[chunk.context_type] = []
            groups[chunk.context_type].append(chunk)
        return groups

    def _determine_processing_order(self, context_groups: Dict[str, List[ContextChunk]]) -> List[str]:
        """Определение оптимального порядка обработки"""
        # Сортировка по приоритету среднего чанка в группе
        context_priorities = []
        for context_type, chunks in context_groups.items():
            avg_priority = sum(chunk.priority for chunk in chunks) / len(chunks)
            context_priorities.append((context_type, avg_priority))

        return [ctx for ctx, _ in sorted(context_priorities, key=lambda x: x[1], reverse=True)]

    async def _process_context_group_parallel(self, context_type: str, chunks: List[ContextChunk],
                                              assessment_id: str) -> Dict[str, Any]:
        """Параллельная обработка группы чанков"""

        if len(chunks) == 1:
            return await self._process_single_chunk_with_retry(chunks[0], assessment_id)

        # Параллельная обработка множественных чанков
        tasks = []
        for chunk in chunks:
            task = asyncio.create_task(
                self._process_single_chunk_with_retry(chunk, assessment_id)
            )
            tasks.append(task)

        # Ожидание завершения всех задач
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обработка результатов и исключений
        successful_results = []
        failed_results = []

        for i, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                failed_results.append({
                    "chunk_id": chunks[i].chunk_id,
                    "error": str(result)
                })
            else:
                successful_results.append(result)

        # Агрегация результатов
        if successful_results:
            aggregated = await self._aggregate_chunk_results_advanced(
                context_type, successful_results, assessment_id
            )
            if failed_results:
                aggregated["failed_chunks"] = failed_results
            return aggregated
        else:
            return {
                "error": "Все чанки завершились с ошибками",
                "failed_chunks": failed_results
            }

    async def _process_single_chunk_with_retry(self, chunk: ContextChunk, assessment_id: str) -> Dict[str, Any]:
        """Обработка одного чанка с умными повторами"""

        # Проверка кэша
        cache_key = f"{chunk.chunk_id}_{hash(chunk.content[:100])}"
        if cache_key in self.processing_cache:
            self.logger.bind_context(assessment_id, "llm_orchestrator").debug(
                f"Используем кэшированный результат для {chunk.chunk_id}"
            )
            return self.processing_cache[cache_key]

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Выбор стратегии повтора
                if attempt == 0:
                    result = await self._process_chunk_standard(chunk, assessment_id)
                elif attempt == 1:
                    result = await self._process_chunk_with_modified_prompt(chunk, assessment_id)
                else:
                    result = await self._process_chunk_with_splitting(chunk, assessment_id)

                # Кэшируем успешный результат
                self.processing_cache[cache_key] = result
                return result

            except Exception as e:
                last_error = e
                self.logger.bind_context(assessment_id, "llm_orchestrator").warning(
                    f"Попытка {attempt + 1} для {chunk.chunk_id} неудачна: {e}"
                )

                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))

        # Все попытки неудачны
        return {
            "chunk_id": chunk.chunk_id,
            "error": str(last_error),
            "context_type": chunk.context_type,
            "attempts_made": max_retries
        }

    async def _process_chunk_standard(self, chunk: ContextChunk, assessment_id: str) -> Dict[str, Any]:
        """Стандартная обработка чанка"""
        prompt = self._create_context_specific_prompt(chunk)

        response = await self.agent.call_llm_structured(
            data_to_analyze=chunk.content,
            extraction_prompt=prompt,
            assessment_id=assessment_id,
            expected_format="JSON"
        )

        return {
            "chunk_id": chunk.chunk_id,
            "context_type": chunk.context_type,
            "analysis": response,
            "metadata": chunk.metadata,
            "processing_method": "standard"
        }

    async def _process_chunk_with_modified_prompt(self, chunk: ContextChunk, assessment_id: str) -> Dict[str, Any]:
        """Обработка с модифицированным промптом"""
        # Создаем более простой промпт для сложных случаев
        simplified_prompt = self._create_simplified_prompt(chunk)

        response = await self.agent.call_llm_structured(
            data_to_analyze=chunk.content[:self.agent.chunker.max_chunk_size // 2],  # Уменьшаем размер
            extraction_prompt=simplified_prompt,
            assessment_id=assessment_id,
            expected_format="JSON"
        )

        return {
            "chunk_id": chunk.chunk_id,
            "context_type": chunk.context_type,
            "analysis": response,
            "metadata": {**chunk.metadata, "simplified": True},
            "processing_method": "simplified"
        }

    async def _process_chunk_with_splitting(self, chunk: ContextChunk, assessment_id: str) -> Dict[str, Any]:
        """Обработка с разбиением чанка"""
        # Разбиваем чанк на более мелкие части
        sub_chunks = self._split_chunk_content(chunk)

        sub_results = []
        for i, sub_content in enumerate(sub_chunks):
            try:
                sub_prompt = self._create_partial_analysis_prompt(chunk, i, len(sub_chunks))

                sub_response = await self.agent.call_llm_structured(
                    data_to_analyze=sub_content,
                    extraction_prompt=sub_prompt,
                    assessment_id=assessment_id,
                    expected_format="JSON"
                )

                sub_results.append(sub_response)

            except Exception as e:
                sub_results.append({"error": str(e), "part": i})

        # Объединяем результаты
        combined_analysis = self._combine_sub_results(sub_results)

        return {
            "chunk_id": chunk.chunk_id,
            "context_type": chunk.context_type,
            "analysis": combined_analysis,
            "metadata": {**chunk.metadata, "split_processed": True, "sub_parts": len(sub_chunks)},
            "processing_method": "split"
        }

    def _create_context_specific_prompt(self, chunk: ContextChunk) -> str:
        """Создание контекстно-специфичного промпта"""
        base_prompts = base_prompts_profiler

        base_prompt = base_prompts.get(chunk.context_type, "Проанализируй предоставленные данные.")

        # Добавляем подсказки обработки если они есть
        processing_hints = ""
        if chunk.processing_hints:
            hints_text = ", ".join(chunk.processing_hints)
            processing_hints = f"\n\nПОДСКАЗКИ ОБРАБОТКИ: {hints_text}"

        return f"""
        Контекст анализа: {chunk.context_type}
        Приоритет: {chunk.priority}/10

        {base_prompt}{processing_hints}

        Отвечай только валидным JSON без дополнительного текста.
        """

    def _create_simplified_prompt(self, chunk: ContextChunk) -> str:
        """Создание упрощенного промпта"""
        return f"""
        Проанализируй данные и извлеки ключевую информацию в JSON формате.
        Контекст: {chunk.context_type}

        Формат ответа:
        {{
            "summary": "краткое описание",
            "key_points": ["ключевой пункт 1", "ключевой пункт 2"],
            "type": "тип контента",
            "importance": "высокая/средняя/низкая"
        }}
        """

    def _create_partial_analysis_prompt(self, chunk: ContextChunk, part_num: int, total_parts: int) -> str:
        """Создание промпта для частичного анализа"""
        return f"""
        Проанализируй часть {part_num + 1} из {total_parts} для контекста {chunk.context_type}.

        Извлеки релевантную информацию в JSON формате:
        {{
            "part_summary": "краткое описание этой части",
            "extracted_data": {{}},
            "part_number": {part_num + 1},
            "continuation_needed": true/false
        }}
        """

    def _split_chunk_content(self, chunk: ContextChunk) -> List[str]:
        """Разбиение содержимого чанка на части"""
        content = chunk.content
        max_part_size = len(content) // 3  # Разбиваем на 3 части

        # Пытаемся разбить по логическим границам
        sections = content.split('\n\n')
        parts = []
        current_part = []
        current_size = 0

        for section in sections:
            section_size = len(section)
            if current_size + section_size <= max_part_size or not current_part:
                current_part.append(section)
                current_size += section_size
            else:
                parts.append('\n\n'.join(current_part))
                current_part = [section]
                current_size = section_size

        if current_part:
            parts.append('\n\n'.join(current_part))

        return parts

    def _combine_sub_results(self, sub_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Объединение результатов частичного анализа"""
        combined = {
            "combined_analysis": True,
            "sub_parts_count": len(sub_results),
            "summary": "",
            "extracted_data": {},
            "key_points": []
        }

        for result in sub_results:
            if "error" not in result:
                if "part_summary" in result:
                    combined["summary"] += result["part_summary"] + " "
                if "extracted_data" in result:
                    combined["extracted_data"].update(result["extracted_data"])
                if "key_points" in result:
                    combined["key_points"].extend(result.get("key_points", []))

        return combined

    async def _aggregate_chunk_results_advanced(self, context_type: str, chunk_results: List[Dict[str, Any]],
                                                assessment_id: str) -> Dict[str, Any]:
        """Продвинутая агрегация результатов"""

        if not chunk_results:
            return {"error": "Нет результатов для агрегации"}

        # Сбор всех данных для умной агрегации
        aggregation_data = {
            "context_type": context_type,
            "total_chunks": len(chunk_results),
            "successful_chunks": len([r for r in chunk_results if "error" not in r]),
            "analyses": []
        }

        for result in chunk_results:
            if "analysis" in result and "error" not in result:
                analysis_data = result["analysis"]
                analysis_data["source_chunk"] = result["chunk_id"]
                analysis_data["processing_method"] = result.get("processing_method", "unknown")
                aggregation_data["analyses"].append(analysis_data)

        if not aggregation_data["analyses"]:
            return {"error": "Нет успешных анализов для агрегации"}

        # Интеллектуальная агрегация с контекстным промптом
        aggregation_prompt = self._create_aggregation_prompt(context_type, len(aggregation_data["analyses"]))

        try:
            aggregated_content = self._format_aggregation_content(aggregation_data["analyses"])

            aggregated_result = await self.agent.call_llm_structured(
                data_to_analyze=aggregated_content,
                extraction_prompt=aggregation_prompt,
                assessment_id=assessment_id,
                expected_format="JSON"
            )

            return {
                "context_type": context_type,
                "aggregated_analysis": aggregated_result,
                "metadata": {
                    "total_chunks": aggregation_data["total_chunks"],
                    "successful_chunks": aggregation_data["successful_chunks"],
                    "aggregation_timestamp": datetime.now().isoformat(),
                    "aggregation_quality": self._assess_aggregation_quality(aggregated_result)
                }
            }

        except Exception as e:
            self.logger.bind_context(assessment_id, "llm_orchestrator").error(
                f"Ошибка агрегации для {context_type}: {e}"
            )

            # Fallback агрегация
            return {
                "context_type": context_type,
                "fallback_aggregation": self._create_fallback_aggregation(aggregation_data["analyses"]),
                "metadata": {
                    "total_chunks": aggregation_data["total_chunks"],
                    "successful_chunks": aggregation_data["successful_chunks"],
                    "fallback_used": True,
                    "error": str(e)
                }
            }

    def _create_aggregation_prompt(self, context_type: str, num_analyses: int) -> str:
        """Создание промпта для агрегации"""
        return f"""
        Агрегируй и синтезируй результаты {num_analyses} анализов для контекста "{context_type}".

        Создай единое связное представление, объединив информацию из всех источников:
        1. Устрани дублирование
        2. Разреши противоречия
        3. Синтезируй дополняющую информацию
        4. Сохрани все важные детали
        5. Создай логическую структуру

        Результат должен быть более полным и точным, чем любой отдельный анализ.
        Отвечай только валидным JSON.
        """

    def _format_aggregation_content(self, analyses: List[Dict[str, Any]]) -> str:
        """Форматирование контента для агрегации"""
        formatted_sections = []

        for i, analysis in enumerate(analyses):
            source_chunk = analysis.get("source_chunk", f"chunk_{i}")
            method = analysis.get("processing_method", "unknown")

            # Удаляем служебные поля
            clean_analysis = {k: v for k, v in analysis.items()
                              if k not in ["source_chunk", "processing_method"]}

            section = f"=== Анализ {i + 1} (из {source_chunk}, метод: {method}) ===\n"
            section += json.dumps(clean_analysis, ensure_ascii=False, indent=2)
            formatted_sections.append(section)

        return "\n\n".join(formatted_sections)

    def _assess_aggregation_quality(self, aggregated_result: Dict[str, Any]) -> float:
        """Оценка качества агрегации"""
        quality_score = 0.0

        # Проверяем полноту
        if isinstance(aggregated_result, dict) and len(aggregated_result) > 0:
            quality_score += 0.3

        # Проверяем структурированность
        if any(isinstance(v, (list, dict)) for v in aggregated_result.values()):
            quality_score += 0.2

        # Проверяем содержательность
        text_content = str(aggregated_result)
        if len(text_content) > 100:
            quality_score += 0.3

        # Проверяем отсутствие ошибок
        if "error" not in str(aggregated_result).lower():
            quality_score += 0.2

        return min(quality_score, 1.0)

    def _create_fallback_aggregation(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Создание fallback агрегации"""
        fallback = {
            "fallback_aggregation": True,
            "combined_data": {},
            "all_analyses": analyses,
            "summary": "Автоматическая агрегация из-за ошибки LLM"
        }

        # Простое объединение данных
        for analysis in analyses:
            for key, value in analysis.items():
                if key not in ["source_chunk", "processing_method"]:
                    if key not in fallback["combined_data"]:
                        fallback["combined_data"][key] = []
                    fallback["combined_data"][key].append(value)

        return fallback


class OutputGenerator:
    """Улучшенный генератор выходных форматов"""

    def __init__(self, logger):
        self.logger = logger
        self.template_cache = {}

    async def generate_outputs(self, agent_profile: AgentProfile, llm_results: Dict[str, Any],
                               processing_stages: List[ProcessingStage], assessment_id: str) -> Dict[str, str]:
        """Генерация всех выходных форматов с улучшенным качеством"""

        outputs = {}

        try:
            # Подготовка контекста для генерации
            generation_context = self._prepare_generation_context(
                agent_profile, llm_results, processing_stages, assessment_id
            )

            # Параллельная генерация выходов
            generation_tasks = [
                self._generate_summary_report_async(generation_context),
                self._generate_architecture_graph_async(generation_context),
                self._generate_detailed_json_async(generation_context),
                self._generate_processing_log_async(processing_stages)
            ]

            results = await asyncio.gather(*generation_tasks, return_exceptions=True)

            output_names = ['summary_report', 'architecture_graph', 'detailed_json', 'processing_log']

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.bind_context(assessment_id, "output_generator").error(
                        f"Ошибка генерации {output_names[i]}: {result}"
                    )
                    outputs[output_names[i]] = f"Ошибка генерации: {str(result)}"
                else:
                    outputs[output_names[i]] = result

        except Exception as e:
            self.logger.bind_context(assessment_id, "output_generator").error(
                f"Критическая ошибка генерации: {e}"
            )

        return outputs

    def _prepare_generation_context(self, agent_profile: AgentProfile, llm_results: Dict[str, Any],
                                    processing_stages: List[ProcessingStage], assessment_id: str) -> Dict[str, Any]:
        """Подготовка контекста для генерации"""
        return {
            "agent_profile": agent_profile,
            "llm_results": llm_results,
            "processing_stages": processing_stages,
            "assessment_id": assessment_id,
            "generation_time": datetime.now(),
            "total_processing_time": sum(
                (stage.end_time - stage.start_time).total_seconds()
                for stage in processing_stages
                if stage.end_time
            )
        }

    def _clean_risk_assessment_fields(self, llm_results: Dict[str, Any]) -> Dict[str, Any]:
        """Очистка полей с оценками рисков из результатов LLM"""
        cleaned_results = {}

        risk_fields_to_remove = {
            "probability_score", "impact_score", "total_score", "risk_level",
            "probability_reasoning", "impact_reasoning", "confidence_level",
            "quality_score", "is_acceptable"
        }

        for context_type, context_result in llm_results.items():
            if isinstance(context_result, dict):
                cleaned_context = {}

                # Копируем основную структуру
                for key, value in context_result.items():
                    if key == "aggregated_analysis" and isinstance(value, dict):
                        # Очищаем вложенный analysis от полей оценки рисков
                        cleaned_analysis = {
                            k: v for k, v in value.items()
                            if k not in risk_fields_to_remove
                        }
                        cleaned_context[key] = cleaned_analysis
                    elif key not in risk_fields_to_remove:
                        cleaned_context[key] = value

                cleaned_results[context_type] = cleaned_context
            else:
                cleaned_results[context_type] = context_result

        return cleaned_results

    async def _generate_detailed_json_async(self, context: Dict[str, Any]) -> str:
        """Генерация детального JSON отчета"""
        agent_profile = context["agent_profile"]
        llm_results = context["llm_results"]
        processing_stages = context["processing_stages"]

        cleaned_llm_results = self._clean_risk_assessment_fields(llm_results)

        detailed_report = {
            "metadata": {
                "assessment_id": context["assessment_id"],
                "generated_at": datetime.now().isoformat(),
                "profiler_version": "2.0.0",
                "total_processing_time": context["total_processing_time"]
            },
            "agent_profile": self._serialize_agent_profile(agent_profile),
            "llm_analysis_results": llm_results,
            "processing_stages": [self._serialize_processing_stage(stage) for stage in processing_stages],
            "analysis_summary": {
                "contexts_analyzed": list(llm_results.keys()),
                "successful_contexts": len([r for r in llm_results.values()
                                            if not isinstance(r, dict) or "error" not in r]),
                "failed_contexts": len([r for r in llm_results.values()
                                        if isinstance(r, dict) and "error" in r]),
                #"data_quality_score": self._calculate_data_quality_score(agent_profile, llm_results)
            },
            #"recommendations": self._generate_recommendations(agent_profile, llm_results)
        }

        return json.dumps(detailed_report, ensure_ascii=False, indent=2, default=str)

    async def _generate_summary_report_async(self, context: Dict[str, Any]) -> str:
        """Асинхронная генерация итогового отчета"""
        agent_profile = context["agent_profile"]
        llm_results = context["llm_results"]
        processing_stages = context["processing_stages"]
        assessment_id = context["assessment_id"]
        total_time = context["total_processing_time"]

        # Используем шаблонизацию для лучшего форматирования
        report_template = self._get_report_template()

        # Подготовка данных для шаблона
        template_data = {
            "assessment_id": assessment_id,
            "generation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "total_time": f"{total_time:.2f}",
            "agent_name": agent_profile.name,
            "agent_type": agent_profile.agent_type.value,
            "agent_version": agent_profile.version,
            "autonomy_level": agent_profile.autonomy_level.value,
            "description": agent_profile.description,
            "target_audience": agent_profile.target_audience,
            "llm_model": agent_profile.llm_model,
            "data_types": ', '.join([str(da.value) for da in agent_profile.data_access]),
            "external_apis": ', '.join(
                agent_profile.external_apis) if agent_profile.external_apis else 'Не используются',
            "operations_per_hour": agent_profile.operations_per_hour or 'Не указано',
            "revenue_per_operation": f"{agent_profile.revenue_per_operation} руб." if agent_profile.revenue_per_operation else 'Не указано',
            "system_prompts_count": len(agent_profile.system_prompts),
            "guardrails_count": len(agent_profile.guardrails),
            "system_prompts": self._format_prompts_list(agent_profile.system_prompts),
            "guardrails": self._format_prompts_list(agent_profile.guardrails),
            "detailed_summary": self._format_detailed_summary(agent_profile.detailed_summary),
            "llm_analysis": self._format_llm_analysis(llm_results),
            "processing_stats": self._format_processing_stats(processing_stages),
            #"data_quality_score": self._calculate_data_quality_score(agent_profile, llm_results)
        }

        return report_template.format(**template_data)

    def _get_report_template(self) -> str:
        """Получение шаблона отчета"""
        if "summary_report" not in self.template_cache:
            self.template_cache["summary_report"] = summary_report_prompt

        return self.template_cache["summary_report"]

    def _format_prompts_list(self, prompts: List[str]) -> str:
        """Форматирование списка промптов"""
        if not prompts:
            return "Промпты не найдены"

        formatted = []
        for i, prompt in enumerate(prompts[:5], 1):  # Показываем только первые 5
            truncated = prompt[:300] + "..." if len(prompt) > 300 else prompt
            formatted.append(f"{i}. {truncated}")

        if len(prompts) > 5:
            formatted.append(f"... и еще {len(prompts) - 5} промптов")

        return "\n".join(formatted)

    def _format_detailed_summary(self, detailed_summary: Optional[Dict[str, str]]) -> str:
        """Форматирование детального саммари"""
        if not detailed_summary:
            return "## 📊 ДЕТАЛЬНЫЙ АНАЛИЗ\n\nДетальное саммари не создано"

        formatted_sections = ["## 📊 ДЕТАЛЬНЫЙ АНАЛИЗ\n"]

        section_titles = {
            'overview': 'ОБЗОР АГЕНТА',
            'technical_architecture': 'ТЕХНИЧЕСКАЯ АРХИТЕКТУРА',
            'operational_model': 'ОПЕРАЦИОННАЯ МОДЕЛЬ',
            'risk_analysis': 'АНАЛИЗ РИСКОВ',
            'security_recommendations': 'РЕКОМЕНДАЦИИ ПО БЕЗОПАСНОСТИ',
            'conclusions': 'ВЫВОДЫ'
        }

        for section_key, section_content in detailed_summary.items():
            title = section_titles.get(section_key, section_key.replace('_', ' ').title())
            formatted_sections.append(f"### {title}\n{section_content}\n")

        return "\n".join(formatted_sections)

    def _format_llm_analysis(self, llm_results: Dict[str, Any]) -> str:
        """Форматирование результатов LLM анализа"""
        if not llm_results:
            return "## 🧠 РЕЗУЛЬТАТЫ LLM АНАЛИЗА\n\nLLM анализ не выполнен"

        sections = ["## 🧠 РЕЗУЛЬТАТЫ LLM АНАЛИЗА\n"]

        for context_type, context_result in llm_results.items():
            context_title = context_type.replace('_', ' ').title()
            sections.append(f"### {context_title}")

            if isinstance(context_result, dict) and 'aggregated_analysis' in context_result:
                analysis = context_result['aggregated_analysis']
                # Создаем краткое представление вместо полного JSON
                summary = self._summarize_analysis(analysis)
                sections.append(f"{summary}\n")
            elif isinstance(context_result, dict) and 'error' in context_result:
                sections.append(f"❌ Ошибка анализа: {context_result['error']}\n")
            else:
                sections.append("Данные анализа доступны в детальном JSON отчете\n")

        return "\n".join(sections)

    def _summarize_analysis(self, analysis: Dict[str, Any]) -> str:
        """Создание краткого саммари анализа"""
        if not isinstance(analysis, dict):
            return str(analysis)[:200] + "..."

        summary_parts = []

        # Ищем ключевые поля для саммари
        key_fields = ['summary', 'description', 'key_points', 'main_findings', 'overview']

        for field in key_fields:
            if field in analysis:
                value = analysis[field]
                if isinstance(value, str) and len(value) > 10:
                    summary_parts.append(value[:150] + "..." if len(value) > 150 else value)
                    break
                elif isinstance(value, list) and value:
                    summary_parts.append(f"Ключевые моменты: {', '.join(map(str, value[:3]))}")
                    break

        if not summary_parts and analysis:
            # Если не нашли стандартные поля, берем первое непустое значение
            for key, value in analysis.items():
                if isinstance(value, str) and len(value) > 10:
                    summary_parts.append(f"{key}: {value[:100]}...")
                    break

        return summary_parts[0] if summary_parts else "Анализ выполнен"

    def _format_processing_stats(self, processing_stages: List[ProcessingStage]) -> str:
        """Форматирование статистики обработки"""
        if not processing_stages:
            return "## 📈 СТАТИСТИКА ОБРАБОТКИ\n\nДанные обработки недоступны"

        total_time = sum(
            (stage.end_time - stage.start_time).total_seconds()
            for stage in processing_stages
            if stage.end_time
        )

        successful_stages = [s for s in processing_stages if s.status == 'completed']
        failed_stages = [s for s in processing_stages if s.status == 'failed']

        stats = [
            "## 📈 СТАТИСТИКА ОБРАБОТКИ\n",
            f"**Всего стадий:** {len(processing_stages)}",
            f"**Успешных:** {len(successful_stages)}",
            f"**Неудачных:** {len(failed_stages)}",
            f"**Общее время:** {total_time:.2f} секунд",
            f"**Среднее время на стадию:** {total_time / len(processing_stages):.2f} секунд\n"
        ]

        # Детали по стадиям
        if len(processing_stages) <= 10:  # Показываем детали только если стадий немного
            stats.append("### Детали выполнения:")
            for stage in processing_stages:
                duration = (stage.end_time - stage.start_time).total_seconds() if stage.end_time else 0
                status_icon = "✅" if stage.status == 'completed' else "❌" if stage.status == 'failed' else "⏳"
                stats.append(f"- {status_icon} {stage.stage_name}: {duration:.2f}с")

        return "\n".join(stats)

    def _prepare_template_data(self, agent_profile: AgentProfile, llm_results: Dict[str, Any],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Подготовка данных для шаблона"""
        return {
            "agent_name": agent_profile.name,
            "agent_type": agent_profile.agent_type.value,
            "agent_version": agent_profile.version,
            "autonomy_level": agent_profile.autonomy_level.value,
            #"data_quality_score": self._calculate_data_quality_score(agent_profile, llm_results),
            "generation_date": context.get("generation_time", datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
            "system_prompts_count": len(agent_profile.system_prompts),
            "guardrails_count": len(agent_profile.guardrails),
            "external_apis": ', '.join(
                agent_profile.external_apis) if agent_profile.external_apis else 'Не используются',
            "operations_per_hour": agent_profile.operations_per_hour or 'Не указано',
            "revenue_per_operation": f"{agent_profile.revenue_per_operation} руб." if agent_profile.revenue_per_operation else 'Не указано',
            "description": agent_profile.description,
            "target_audience": agent_profile.target_audience
        }

    async def _generate_architecture_graph_async(self, context: Dict[str, Any]) -> str:
        """Асинхронная генерация диаграммы архитектуры"""
        agent_profile = context["agent_profile"]
        llm_results = context["llm_results"]

        # ИСПРАВЛЕНО: Создаем template_data
        template_data = {
            "agent_name": agent_profile.name,
            "agent_type": agent_profile.agent_type.value if hasattr(agent_profile.agent_type, 'value') else str(
                agent_profile.agent_type),
            "agent_version": agent_profile.version,
            "autonomy_level": agent_profile.autonomy_level.value if hasattr(agent_profile.autonomy_level,
                                                                            'value') else str(
                agent_profile.autonomy_level),
            #"data_quality_score": self._calculate_data_quality_score(agent_profile, llm_results),
            "generation_date": context.get("generation_time", datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
            "system_prompts_count": len(agent_profile.system_prompts),
            "guardrails_count": len(agent_profile.guardrails),
            "external_apis": ', '.join(
                agent_profile.external_apis) if agent_profile.external_apis else 'Не используются',
            "operations_per_hour": agent_profile.operations_per_hour or 'Не указано',
            "revenue_per_operation": f"{agent_profile.revenue_per_operation} руб." if agent_profile.revenue_per_operation else 'Не указано',
            "description": agent_profile.description,
            "target_audience": agent_profile.target_audience
        }

        # Создаем более детальную диаграмму на основе собранных данных
        mermaid_graph = self._build_comprehensive_architecture_graph(agent_profile, llm_results, template_data)

        return mermaid_graph

    def _build_comprehensive_architecture_graph(self, agent_profile: AgentProfile, llm_results: Dict[str, Any],
                                                template_data: Dict[str, Any]) -> str:
        """Build a comprehensive architecture diagram for an AI agent, optimized for LLM interpretability and risk assessment."""

        # ИСПРАВЛЕНО: Проверяем наличие template_data
        if not template_data:
            # Создаем базовые template_data если они отсутствуют
            template_data = {
                "agent_name": agent_profile.name,
                "agent_type": agent_profile.agent_type.value if hasattr(agent_profile.agent_type, 'value') else str(
                    agent_profile.agent_type),
                "agent_version": agent_profile.version,
                "autonomy_level": agent_profile.autonomy_level.value if hasattr(agent_profile.autonomy_level,
                                                                                'value') else str(
                    agent_profile.autonomy_level),
                #"data_quality_score": self._calculate_data_quality_score(agent_profile, llm_results),
                "generation_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "system_prompts_count": len(agent_profile.system_prompts),
                "guardrails_count": len(agent_profile.guardrails),
                "external_apis": ', '.join(
                    agent_profile.external_apis) if agent_profile.external_apis else 'Не используются',
                "operations_per_hour": agent_profile.operations_per_hour or 'Не указано',
                "revenue_per_operation": f"{agent_profile.revenue_per_operation} руб." if agent_profile.revenue_per_operation else 'Не указано',
                "description": agent_profile.description,
                "target_audience": agent_profile.target_audience
            }

        graph_lines = [
            "%% Comprehensive architecture diagram for AI Agent",
            f"%% Agent: {template_data['agent_name']} (Type: {template_data['agent_type']}, Version: {template_data['agent_version']})",
            f"%% Autonomy Level: {template_data['autonomy_level']}",
            f"%% Generated: {template_data['generation_date']}",
            "graph TD",
            "    A[Пользователь] -->|Ввод запроса| B[ИИ-Агент]",
            "    B -->|Маршрутизация| C{Обработка запроса}"
        ]

        node_counter = ord('D')

        # Add system prompts with context
        if agent_profile.system_prompts:
            prompt_node = chr(node_counter)
            graph_lines.append(f"    %% System Prompts: Defines agent behavior and constraints")
            graph_lines.append(
                f"    C -->|Использует {template_data['system_prompts_count']} промптов| {prompt_node}[Системные промпты]")
            graph_lines.append(f"    {prompt_node} -->|Передача контекста| E[LLM]")
            node_counter += 1
        else:
            # Если нет системных промптов, прямая связь с LLM
            graph_lines.append(f"    C -->|Прямой вызов| E[LLM]")

        # Add external APIs with risk annotation
        if agent_profile.external_apis and template_data['external_apis'] != 'Не используются':
            api_node = chr(node_counter)
            graph_lines.append(f"    %% External APIs: Potential risk points for data privacy and reliability")
            graph_lines.append(f"    C -->|Интеграция| {api_node}[🔌 Внешние API]")
            for i, api in enumerate(agent_profile.external_apis[:3], 1):
                api_sub_node = f"{api_node}{i}"
                graph_lines.append(f"    {api_node} -->|API вызов| {api_sub_node}[{api}]")
            node_counter += 1

        # Add data access with data quality annotation
        if agent_profile.data_access:
            data_node = chr(node_counter)
            graph_lines.append(f"    C -->|Доступ к данным| {data_node}[Данные]")
            for i, data_type in enumerate(agent_profile.data_access[:3], 1):
                data_sub_node = f"{data_node}{i}"
                data_value = data_type.value if hasattr(data_type, 'value') else str(data_type)
                graph_lines.append(f"    {data_node} -->|Тип данных| {data_sub_node}[{data_value}]")
            node_counter += 1

        # Add guardrails with risk mitigation annotation
        if agent_profile.guardrails:
            guard_node = chr(node_counter)
            graph_lines.append(f"    %% Guardrails: Mitigate risks, {template_data['guardrails_count']} configured")
            graph_lines.append(f"    E -->|Фильтрация| {guard_node}[Guardrails]")
            graph_lines.append(f"    {guard_node} -->|Безопасный вывод| H[Ответ пользователю]")
            node_counter += 1
        else:
            graph_lines.append(f"    %% No Guardrails: Potential risk point")
            graph_lines.append(f"    E -->|Прямой вывод| H[Ответ пользователю]")

        # Add performance and operational context
        graph_lines.extend([
            "",
            f"    %% Operational Stats: {template_data['operations_per_hour']} ops/hour, Revenue: {template_data['revenue_per_operation']}",
            f"    %% Description: {template_data['description'][:100]}...",
            f"    %% Target Audience: {template_data['target_audience']}",
        ])

        # Add styling for visual clarity
        graph_lines.extend([
            "",
            "    %% Styling for visual distinction",
            "    classDef userClass fill:#e1f5fe,stroke:#0288d1,stroke-width:2px",
            "    classDef agentClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px",
            "    classDef llmClass fill:#e8f5e8,stroke:#388e3c,stroke-width:2px",
            "    classDef apiClass fill:#fff3e0,stroke:#f57c00,stroke-width:2px",
            "    classDef dataClass fill:#fce4ec,stroke:#d81b60,stroke-width:2px",
            "    classDef guardClass fill:#ffebee,stroke:#c62828,stroke-width:2px",
            "",
            "    class A userClass",
            "    class B,C agentClass",
            "    class E llmClass",
            "    class H userClass"
        ])

        # Apply styling to dynamic nodes
        if agent_profile.external_apis and node_counter > ord('D'):
            api_node = chr(ord('D'))
            graph_lines.append(f"    class {api_node} apiClass")
            for i in range(1, min(len(agent_profile.external_apis) + 1, 4)):
                graph_lines.append(f"    class {api_node}{i} apiClass")

        if agent_profile.data_access and node_counter > ord('E'):
            data_node = chr(ord('E') if not agent_profile.external_apis else ord('F'))
            graph_lines.append(f"    class {data_node} dataClass")
            for i in range(1, min(len(agent_profile.data_access) + 1, 4)):
                graph_lines.append(f"    class {data_node}{i} dataClass")

        if agent_profile.guardrails:
            guard_node_char = chr(node_counter - 1)
            graph_lines.append(f"    class {guard_node_char} guardClass")

        return "\n".join(graph_lines)

    def _serialize_agent_profile(self, agent_profile: AgentProfile) -> Dict[str, Any]:
        """Сериализация профиля агента"""
        return {
            "name": agent_profile.name,
            "description": agent_profile.description,
            "agent_type": agent_profile.agent_type.value,
            "llm_model": agent_profile.llm_model,
            "autonomy_level": agent_profile.autonomy_level.value,
            "data_access": [da.value for da in agent_profile.data_access],
            "external_apis": agent_profile.external_apis,
            "target_audience": agent_profile.target_audience,
            "operations_per_hour": agent_profile.operations_per_hour,
            "revenue_per_operation": agent_profile.revenue_per_operation,
            "system_prompts": agent_profile.system_prompts,
            "guardrails": agent_profile.guardrails,
            "source_files": agent_profile.source_files,
            "detailed_summary": agent_profile.detailed_summary,
            "version": agent_profile.version,
            "created_at": agent_profile.created_at.isoformat() if agent_profile.created_at else None,
            "updated_at": agent_profile.updated_at.isoformat() if agent_profile.updated_at else None
        }

    def _serialize_processing_stage(self, stage: ProcessingStage) -> Dict[str, Any]:
        """Сериализация стадии обработки"""
        return {
            "stage_name": stage.stage_name,
            "start_time": stage.start_time.isoformat(),
            "end_time": stage.end_time.isoformat() if stage.end_time else None,
            "status": stage.status,
            "execution_time": (stage.end_time - stage.start_time).total_seconds() if stage.end_time else None,
            "output_files": stage.output_files or [],
            "metrics": stage.metrics or {},
            "error_details": stage.error_details
        }

    def _generate_recommendations(self, agent_profile: AgentProfile, llm_results: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций на основе анализа"""
        recommendations = []

        # Анализ системных промптов
        if not agent_profile.system_prompts:
            recommendations.append("Добавить системные промпты для лучшего контроля поведения агента")
        elif len(agent_profile.system_prompts) > 5:
            recommendations.append("Рассмотрить консолидацию системных промптов для упрощения управления")

        # Анализ ограничений
        if not agent_profile.guardrails:
            recommendations.append("Внедрить guardrails для повышения безопасности агента")

        # Анализ автономности
        if agent_profile.autonomy_level == AutonomyLevel.AUTONOMOUS:
            recommendations.append("Рассмотреть добавление дополнительного контроля для автономного агента")

        # Анализ данных
        if DataSensitivity.PERSONAL in agent_profile.data_access or DataSensitivity.FINANCIAL in agent_profile.data_access:
            recommendations.append("Усилить меры защиты персональных и финансовых данных")

        # Анализ внешних API
        if agent_profile.external_apis and len(agent_profile.external_apis) > 3:
            recommendations.append("Провести аудит безопасности внешних интеграций")

        # Анализ документации
        if agent_profile.detailed_summary:
            summary_length = sum(len(section) for section in agent_profile.detailed_summary.values())
            if summary_length < 1000:
                recommendations.append("Расширить документацию агента для лучшего понимания функциональности")
        else:
            recommendations.append("Создать детальную документацию агента")

        return recommendations

    async def _generate_processing_log_async(self, processing_stages: List[ProcessingStage]) -> str:
        """Асинхронная генерация лога обработки"""
        if not processing_stages:
            return "Лог обработки недоступен"

        log_lines = [
            "=== ПРОФИЛИРОВАНИЕ ИИ-АГЕНТА: ЛОГ ОБРАБОТКИ ===",
            f"Время начала: {processing_stages[0].start_time}",
            ""
        ]

        total_time = 0
        successful_stages = 0

        for stage in processing_stages:
            execution_time = (stage.end_time - stage.start_time).total_seconds() if stage.end_time else 0
            total_time += execution_time

            # Начало стадии
            log_lines.append(f"[{stage.start_time.strftime('%H:%M:%S')}] 🔄 НАЧАЛО: {stage.stage_name}")

            # Метрики
            if stage.metrics:
                for metric_name, metric_value in stage.metrics.items():
                    log_lines.append(f"  📊 {metric_name}: {metric_value}")

            # Выходные файлы
            if stage.output_files:
                for output_file in stage.output_files:
                    log_lines.append(f"  📁 Создан: {output_file}")

            # Завершение стадии
            status_icon = "✅" if stage.status == "completed" else "❌" if stage.status == "failed" else "⏳"
            end_time_str = stage.end_time.strftime('%H:%M:%S') if stage.end_time else 'N/A'
            log_lines.append(f"[{end_time_str}] {status_icon} ЗАВЕРШЕНО: {stage.stage_name} ({execution_time:.2f}с)")

            if stage.error_details and stage.status == "failed":
                log_lines.append(f"  ❌ Ошибка: {stage.error_details}")

            log_lines.append("")

            if stage.status == "completed":
                successful_stages += 1

        # Итоговая статистика
        log_lines.extend([
            f"=== ИТОГОВАЯ СТАТИСТИКА ===",
            f"Общее время: {total_time:.2f} секунд",
            f"Успешных стадий: {successful_stages}/{len(processing_stages)}",
            f"Процент успеха: {(successful_stages / len(processing_stages) * 100):.1f}%"
        ])

        return "\n".join(log_lines)

    #def _calculate_data_quality_score(self, agent_profile: AgentProfile, llm_results: Dict[str, Any]) -> float:
    #   """Расчет оценки качества данных"""
    #    score = 0.0
    #    max_score = 100.0

        # Базовая информация (30 баллов)
    #    if agent_profile.name and agent_profile.name != "Unknown":
    #        score += 10
    #    if agent_profile.description and len(agent_profile.description) > 50:
    #        score += 10
    #    if agent_profile.agent_type and str(agent_profile.agent_type) != "other":
    #        score += 10

        # Техническая информация (25 баллов)
    #    if agent_profile.llm_model and agent_profile.llm_model != "unknown":
    #        score += 10
    #    if agent_profile.external_apis:
    #        score += 8
    #    if len(agent_profile.data_access) > 0:
    #        score += 7

        # Промпты и ограничения (25 баллов)
    #    if agent_profile.system_prompts:
    #        score += 15
    #    if agent_profile.guardrails:
    #        score += 10

        # Результаты LLM анализа (20 баллов)
    #    if llm_results:
    #        successful_contexts = len([r for r in llm_results.values() if not isinstance(r, dict) or "error" not in r])
    #        total_contexts = len(llm_results)
    #        if total_contexts > 0:
    #            score += (successful_contexts / total_contexts) * 20

    #    return min(score, max_score)
    def _calculate_data_quality_score(self, agent_profile: AgentProfile, llm_results: Dict[str, Any]) -> Optional[
        float]:
        """ИСПРАВЛЕНО: Возвращаем None вместо расчета оценки качества для профилирования"""
        return None

class EnhancedProfilerAgent(AnalysisAgent):
    """
    Улучшенный профайлер-агент с новой архитектурой
    Реализует flowchart: FileSystemCrawler -> Parsers -> ContextAwareChunker -> LLMOrchestrator -> OutputGenerator
    """

    def __init__(self, config: AgentConfig):
        super().__init__(config)

        # Инициализация компонентов новой архитектуры
        self.fs_crawler = FileSystemCrawler(self.logger)
        self.chunker = ContextAwareChunker(max_chunk_size=6000)
        self.llm_orchestrator = LLMOrchestrator(self)
        self.output_generator = OutputGenerator(self.logger)

        # Классические инструменты как fallback
        self.document_parser = create_document_parser()
        self.code_analyzer = create_code_analyzer()
        self.prompt_analyzer = create_prompt_analyzer()

        # Отслеживание стадий
        self.processing_stages: List[ProcessingStage] = []
        self.current_stage: Optional[ProcessingStage] = None

    def get_system_prompt(self) -> str:
        """Системный промпт для улучшенного профайлера"""
        return profiler_system_prompt

    async def process(self, input_data: Dict[str, Any], assessment_id: str) -> AgentTaskResult:
        """
        Основная обработка с улучшенной архитектурой
        """
        start_time = datetime.now()

        try:
            with LogContext("enhanced_profiling", assessment_id, self.name):
                # Извлечение входных данных
                source_files = input_data.get("source_files", [])
                preliminary_name = input_data.get("agent_name", "Unknown_Agent")

                if not source_files:
                    raise ValueError("Не предоставлены файлы для анализа")

                # Этап 1: Сканирование файловой системы
                await self._start_stage("file_system_crawling", assessment_id)
                files_metadata = await self.fs_crawler.crawl_sources(source_files, max_files=200)
                await self._complete_stage(metrics={
                    "files_found": len(files_metadata),
                    "file_types": len(set(f.file_type.value for f in files_metadata)),
                    "unique_files": len(set(f.content_hash for f in files_metadata if f.content_hash))
                })

                # Этап 2: Парсинг файлов
                await self._start_stage("comprehensive_parsing", assessment_id)
                parsed_data = await self._parse_all_files_optimized(files_metadata, assessment_id)
                await self._complete_stage(metrics={
                    "documents_parsed": len(parsed_data.get("documents", {})),
                    "code_files_parsed": len(parsed_data.get("code_files", {})),
                    "config_files_parsed": len(parsed_data.get("config_files", {})),
                    "prompt_files_parsed": len(parsed_data.get("prompt_files", {}))
                })

                def custom_serializer(obj):
                    """Преобразует неподдерживаемые JSON типы, такие как datetime, в сериализуемый формат."""
                    if isinstance(obj, datetime):
                        return obj.isoformat()  # Преобразует datetime в строку в формате ISO 8601
                    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

                # Предполагается, что parsed_data - это словарь
                with open('parsed_data.txt', 'w', encoding='utf-8') as f:
                    json.dump(parsed_data, f, ensure_ascii=False, indent=4, default=custom_serializer)


                # Этап 3: Контекстно-осознанный чанкинг
                await self._start_stage("context_aware_chunking", assessment_id)
                chunks = await self.chunker.create_chunks(files_metadata, parsed_data)

                output_dir = "chunk_data"
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(output_dir, f"chunks_{assessment_id}_{timestamp}.json")

                chunks_data = [{
                    "chunk_id": i,
                    "context_type": c.context_type,
                    "size_tokens": c.size_tokens,
                    "content": c.content
                } for i, c in enumerate(chunks)]

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks_data, f, ensure_ascii=False, indent=2)

                await self._complete_stage(metrics={
                    "chunks_created": len(chunks),
                    "contexts_identified": len(set(c.context_type for c in chunks)),
                    "avg_chunk_size": sum(c.size_tokens for c in chunks) / len(chunks) if chunks else 0,
                    "output_file": output_file
                })

                # Этап 4: LLM оркестрация
                await self._start_stage("llm_orchestration", assessment_id)
                llm_results = await self.llm_orchestrator.process_chunks(chunks, assessment_id)
                await self._complete_stage(metrics={
                    "contexts_processed": len(llm_results),
                    "successful_contexts": len(
                        [r for r in llm_results.values() if not isinstance(r, dict) or "error" not in r]),
                    "cache_hits": len(self.llm_orchestrator.processing_cache)
                })

                # Этап 5: Создание профиля агента
                await self._start_stage("profile_synthesis", assessment_id)
                agent_profile = await self._create_agent_profile_enhanced(
                    llm_results, parsed_data, preliminary_name, assessment_id
                )
                await self._complete_stage()

                # Этап 6: Генерация выходных форматов
                await self._start_stage("output_generation", assessment_id)
                outputs = await self.output_generator.generate_outputs(
                    agent_profile, llm_results, self.processing_stages, assessment_id
                )
                output_files = await self._save_outputs(outputs, assessment_id)
                await self._complete_stage(output_files=output_files)

                # Создание результата
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()

                return AgentTaskResult(
                    agent_name=self.name,
                    task_type="enhanced_profiling",
                    status=ProcessingStatus.COMPLETED,
                    result_data={
                        "agent_profile": self._serialize_agent_profile_for_result(agent_profile),
                        "llm_analysis_results": llm_results,
                        "processing_stages": [asdict(stage) for stage in self.processing_stages],
                        "output_files": output_files,
                        "performance_metrics": self._calculate_performance_metrics(),
                        #"data_quality_score": self.output_generator._calculate_data_quality_score(agent_profile,llm_results)
                    },
                    start_time=start_time,
                    end_time=end_time,
                    execution_time_seconds=execution_time
                )

        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            # Завершаем текущую стадию с ошибкой
            if self.current_stage and self.current_stage.end_time is None:
                self.current_stage.end_time = end_time
                self.current_stage.status = "failed"
                self.current_stage.error_details = str(e)

            return AgentTaskResult(
                agent_name=self.name,
                task_type="enhanced_profiling",
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time
            )

    async def _start_stage(self, stage_name: str, assessment_id: str):
        """Начало новой стадии обработки"""
        stage = ProcessingStage(
            stage_name=stage_name,
            start_time=datetime.now(),
            status="in_progress"
        )

        self.processing_stages.append(stage)
        self.current_stage = stage

        self.logger.bind_context(assessment_id, self.name).info(
            f"🔄 Начало стадии: {stage_name}"
        )

    async def _complete_stage(self, metrics: Dict[str, Any] = None, output_files: List[str] = None):
        """Завершение текущей стадии"""
        if self.current_stage:
            self.current_stage.end_time = datetime.now()
            self.current_stage.status = "completed"
            self.current_stage.metrics = metrics or {}
            self.current_stage.output_files = output_files or []

            execution_time = (self.current_stage.end_time - self.current_stage.start_time).total_seconds()

            self.logger.bind_context("unknown", self.name).info(
                f"✅ Завершена стадия: {self.current_stage.stage_name} ({execution_time:.2f}с)"
            )

            self.current_stage = None

    async def _parse_all_files_optimized(self, files_metadata: List[FileMetadata], assessment_id: str) -> Dict[
        str, Any]:
        """Оптимизированный парсинг файлов с параллельной обработкой"""

        parsed_data = {
            "documents": {},
            "code_files": {},
            "config_files": {},
            "prompt_files": {},
            "errors": [],
            "parsing_stats": {
                "total_files": len(files_metadata),
                "processed_files": 0,
                "skipped_files": 0,
                "error_files": 0
            }
        }

        # Группировка файлов по типам для оптимизированной обработки
        files_by_type = {}
        for file_meta in files_metadata:
            if file_meta.file_type not in files_by_type:
                files_by_type[file_meta.file_type] = []
            files_by_type[file_meta.file_type].append(file_meta)

        # Параллельная обработка по типам файлов
        parsing_tasks = []

        # Документы
        if FileType.DOCUMENT in files_by_type:
            task = asyncio.create_task(
                self._parse_documents_parallel(files_by_type[FileType.DOCUMENT])
            )
            parsing_tasks.append(("documents", task))

        # Код
        if FileType.CODE in files_by_type:
            task = asyncio.create_task(
                self._parse_code_files_parallel(files_by_type[FileType.CODE])
            )
            parsing_tasks.append(("code_files", task))

        # Конфигурации
        if FileType.CONFIG in files_by_type:
            task = asyncio.create_task(
                self._parse_config_files_parallel(files_by_type[FileType.CONFIG])
            )
            parsing_tasks.append(("config_files", task))

        # Промпты
        if FileType.PROMPT in files_by_type:
            task = asyncio.create_task(
                self._parse_prompt_files_parallel(files_by_type[FileType.PROMPT])
            )
            parsing_tasks.append(("prompt_files", task))

        # Ожидание завершения всех задач
        for data_type, task in parsing_tasks:
            try:
                result = await task
                parsed_data[data_type] = result["data"]
                parsed_data["errors"].extend(result["errors"])
                parsed_data["parsing_stats"]["processed_files"] += result["processed"]
                parsed_data["parsing_stats"]["error_files"] += result["errors_count"]
            except Exception as e:
                self.logger.bind_context(assessment_id, self.name).error(
                    f"Ошибка парсинга {data_type}: {e}"
                )
                parsed_data["errors"].append(f"Критическая ошибка парсинга {data_type}: {e}")

        # Добавляем агрегированные анализы
        if parsed_data["code_files"]:
            parsed_data["code_analysis"] = await self._create_code_summary(parsed_data["code_files"])

        if parsed_data["prompt_files"] or self._has_prompts_in_data(parsed_data):
            parsed_data["prompt_analysis"] = await self._create_prompt_summary(parsed_data)

        return parsed_data

    async def _parse_documents_parallel(self, doc_files: List[FileMetadata]) -> Dict[str, Any]:
        """Параллельный парсинг документов"""
        result = {"data": {}, "errors": [], "processed": 0, "errors_count": 0}

        if not doc_files:
            return result

        # Создаем задачи для парсинга документов
        tasks = []
        for file_meta in doc_files:
            task = asyncio.create_task(
                self._parse_single_document(file_meta)
            )
            tasks.append((file_meta.path, task))

        # Обрабатываем результаты
        for file_path, task in tasks:
            try:
                doc_result = await task
                if doc_result["success"]:
                    result["data"][file_path] = doc_result["data"]
                    result["processed"] += 1
                else:
                    result["errors"].append(f"Ошибка парсинга документа {file_path}: {doc_result['error']}")
                    result["errors_count"] += 1
            except Exception as e:
                result["errors"].append(f"Критическая ошибка документа {file_path}: {e}")
                result["errors_count"] += 1

        return result

    async def _parse_single_document(self, file_meta: FileMetadata) -> Dict[str, Any]:
        """Парсинг одного документа"""
        try:
            parsed_doc = self.document_parser.parse_document(file_meta.path)

            if parsed_doc.success:
                return {
                    "success": True,
                    "data": {
                        "content": parsed_doc.content,
                        "sections": parsed_doc.sections,
                        "tables": parsed_doc.tables,
                        "metadata": parsed_doc.metadata,
                        "file_type": file_meta.file_type.value,
                        "parsing_time": parsed_doc.parsing_time
                    }
                }
            else:
                return {
                    "success": False,
                    "error": parsed_doc.error_message or "Неизвестная ошибка парсинга"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _parse_code_files_parallel(self, code_files: List[FileMetadata]) -> Dict[str, Any]:
        """Параллельный парсинг кода"""
        result = {"data": {}, "errors": [], "processed": 0, "errors_count": 0}

        for file_meta in code_files:
            try:
                with open(file_meta.path, 'r', encoding=file_meta.encoding or 'utf-8') as f:
                    content = f.read()

                result["data"][file_meta.path] = {
                    "content": content,
                    "language": file_meta.language,
                    "size": file_meta.size_bytes,
                    "encoding": file_meta.encoding,
                    "last_modified": file_meta.last_modified.isoformat()
                }
                result["processed"] += 1

            except Exception as e:
                result["errors"].append(f"Ошибка чтения кода {file_meta.path}: {e}")
                result["errors_count"] += 1

        return result

    async def _parse_config_files_parallel(self, config_files: List[FileMetadata]) -> Dict[str, Any]:
        """Параллельный парсинг конфигураций"""
        result = {"data": {}, "errors": [], "processed": 0, "errors_count": 0}

        for file_meta in config_files:
            try:
                with open(file_meta.path, 'r', encoding=file_meta.encoding or 'utf-8') as f:
                    content = f.read()

                # Пытаемся парсить структурированные форматы
                parsed_config = self._parse_structured_config(content, Path(file_meta.path).suffix.lower())

                result["data"][file_meta.path] = {
                    "content": content,
                    "parsed_config": parsed_config,
                    "size": file_meta.size_bytes,
                    "config_type": Path(file_meta.path).suffix.lower()
                }
                result["processed"] += 1

            except Exception as e:
                result["errors"].append(f"Ошибка чтения конфигурации {file_meta.path}: {e}")
                result["errors_count"] += 1

        return result

    async def _parse_prompt_files_parallel(self, prompt_files: List[FileMetadata]) -> Dict[str, Any]:
        """Параллельный парсинг промптов"""
        result = {"data": {}, "errors": [], "processed": 0, "errors_count": 0}

        for file_meta in prompt_files:
            try:
                with open(file_meta.path, 'r', encoding=file_meta.encoding or 'utf-8') as f:
                    content = f.read()

                result["data"][file_meta.path] = {
                    "content": content,
                    "size": file_meta.size_bytes,
                    "prompt_type": "dedicated_prompt_file"
                }
                result["processed"] += 1

            except Exception as e:
                result["errors"].append(f"Ошибка чтения промпта {file_meta.path}: {e}")
                result["errors_count"] += 1

        return result

    def _parse_structured_config(self, content: str, extension: str) -> Optional[Dict[str, Any]]:
        """Парсинг структурированных конфигураций"""
        try:
            if extension == '.json':
                return json.loads(content)
            elif extension in ['.yaml', '.yml']:
                # Простой YAML парсер без внешних зависимостей
                lines = content.split('\n')
                config = {}
                for line in lines:
                    if ':' in line and not line.strip().startswith('#'):
                        key, value = line.split(':', 1)
                        config[key.strip()] = value.strip()
                return config
            elif extension == '.env':
                config = {}
                lines = content.split('\n')
                for line in lines:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.split('=', 1)
                        config[key.strip()] = value.strip()
                return config
        except Exception:
            pass
        return None

    def _has_prompts_in_data(self, parsed_data: Dict[str, Any]) -> bool:
        """Проверка наличия промптов в парсированных данных"""
        # Проверяем документы на наличие промптов
        for doc_data in parsed_data.get("documents", {}).values():
            if isinstance(doc_data, dict):
                content = doc_data.get("content", "").lower()
                if any(keyword in content for keyword in ["prompt", "system", "instruction", "guardrail"]):
                    return True

        # Проверяем код на наличие промптов
        for code_data in parsed_data.get("code_files", {}).values():
            if isinstance(code_data, dict):
                content = code_data.get("content", "").lower()
                if any(keyword in content for keyword in ["system_prompt", "prompt", "instruction"]):
                    return True

        return False

    async def _create_code_summary(self, code_files: Dict[str, Any]) -> Dict[str, Any]:
        """Создание саммари анализа кода"""
        summary = {
            "total_files": len(code_files),
            "languages": {},
            "total_lines": 0,
            "average_file_size": 0,
            "complexity_indicators": []
        }

        for file_path, file_data in code_files.items():
            if isinstance(file_data, dict):
                language = file_data.get("language", "unknown")
                content = file_data.get("content", "")

                # Подсчет языков
                if language not in summary["languages"]:
                    summary["languages"][language] = 0
                summary["languages"][language] += 1

                # Подсчет строк
                lines = len(content.split('\n'))
                summary["total_lines"] += lines

                # Поиск индикаторов сложности
                if "async" in content or "await" in content:
                    summary["complexity_indicators"].append("asynchronous_code")
                if "class" in content:
                    summary["complexity_indicators"].append("object_oriented")
                if any(keyword in content.lower() for keyword in ["ai", "ml", "model", "neural", "llm"]):
                    summary["complexity_indicators"].append("ai_ml_related")

        if code_files:
            summary["average_file_size"] = summary["total_lines"] / len(code_files)

        # Убираем дубликаты
        summary["complexity_indicators"] = list(set(summary["complexity_indicators"]))

        return summary

    def _construct_agent_profile(self, profile_data: Dict[str, Any]) -> AgentProfile:
        """ИСПРАВЛЕННОЕ создание объекта AgentProfile"""

        try:
            # Преобразуем строковые значения в enum'ы
            agent_type = AgentType(profile_data.get("agent_type", "other"))
            autonomy_level = AutonomyLevel(profile_data.get("autonomy_level", "supervised"))

            # Преобразуем data_access в enum'ы
            data_access_list = []
            for da in profile_data.get("data_access", ["internal"]):
                try:
                    data_access_list.append(DataSensitivity(da))
                except ValueError:
                    data_access_list.append(DataSensitivity.INTERNAL)

            return AgentProfile(
                name=profile_data.get("name", "Unknown Agent"),
                version=profile_data.get("version", "1.0"),
                description=profile_data.get("description", "Автоматически сгенерированное описание"),
                agent_type=agent_type,
                llm_model=profile_data.get("llm_model", "unknown"),
                autonomy_level=autonomy_level,
                data_access=data_access_list,
                external_apis=profile_data.get("external_apis", []),
                target_audience=profile_data.get("target_audience", "Общая аудитория"),
                operations_per_hour=profile_data.get("operations_per_hour"),
                revenue_per_operation=profile_data.get("revenue_per_operation"),
                system_prompts=profile_data.get("system_prompts", []),
                guardrails=profile_data.get("guardrails", []),
                source_files=profile_data.get("source_files", []),
                detailed_summary=profile_data.get("detailed_summary"),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

        except Exception as e:
            self.logger.bind_context("unknown", self.name).error(
                f"Ошибка создания AgentProfile: {e}"
            )

            # Fallback профиль
            return AgentProfile(
                name=profile_data.get("name", "Unknown Agent"),
                version="1.0",
                description="Fallback профиль из-за ошибки создания",
                agent_type=AgentType.OTHER,
                llm_model="unknown",
                autonomy_level=AutonomyLevel.SUPERVISED,
                data_access=[DataSensitivity.INTERNAL],
                external_apis=[],
                target_audience="Общая аудитория",
                system_prompts=[],
                guardrails=[],
                source_files=[],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

    async def _create_prompt_summary(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Создание саммари анализа промптов"""
        prompt_sources = []

        # Сбор промптов из всех источников
        for prompt_data in parsed_data.get("prompt_files", {}).values():
            if isinstance(prompt_data, dict):
                prompt_sources.append(prompt_data["content"])

        # Промпты из документов
        for doc_data in parsed_data.get("documents", {}).values():
            if isinstance(doc_data, dict):
                for section_name, section_content in doc_data.get("sections", {}).items():
                    if any(keyword in section_name.lower() for keyword in ["prompt", "instruction", "system"]):
                        prompt_sources.append(section_content)

        # Промпты из кода
        for code_data in parsed_data.get("code_files", {}).values():
            if isinstance(code_data, dict):
                content = code_data.get("content", "")
                # Поиск строковых литералов с промптами
                import re
                prompt_patterns = [
                    r'system_prompt\s*=\s*["\']([^"\']+)["\']',
                    r'prompt\s*=\s*["\']([^"\']+)["\']',
                    r'instruction\s*=\s*["\']([^"\']+)["\']'
                ]
                for pattern in prompt_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    prompt_sources.extend(matches)

        summary = {
            "total_prompts": len(prompt_sources),
            "average_length": sum(len(p) for p in prompt_sources) / len(prompt_sources) if prompt_sources else 0,
            "has_system_prompts": any("system" in p.lower() for p in prompt_sources),
            "has_guardrails": any(keyword in " ".join(prompt_sources).lower() for keyword in
                                  ["don't", "not", "never", "avoid", "restrict"]),
            "complexity_score": self._calculate_prompt_complexity(prompt_sources)
        }

        return summary

    def _calculate_prompt_complexity(self, prompts: List[str]) -> float:
        """Простая оценка сложности промптов"""
        if not prompts:
            return 0.0

        total_complexity = 0
        for prompt in prompts:
            complexity = 0

            # Длина промпта
            complexity += min(len(prompt) / 1000, 2)

            # Количество инструкций
            instruction_count = len(
                [line for line in prompt.split('\n') if line.strip().startswith(('-', '•', '1.', '2.'))])
            complexity += min(instruction_count / 10, 2)

            # Сложные конструкции
            if any(word in prompt.lower() for word in ["if", "when", "unless", "depending"]):
                complexity += 1

            total_complexity += complexity

        return min(total_complexity / len(prompts), 10.0)

    def _create_fallback_llm_result(self, preliminary_name: str) -> Dict[str, Any]:
        """Создание fallback результата LLM"""
        return {
            "name": preliminary_name or "Unknown Agent",
            "version": "1.0",
            "description": f"ИИ-агент {preliminary_name}. Автоматически сгенерированное описание из-за ошибки анализа LLM.",
            "agent_type": "other",
            "llm_model": "unknown",
            "autonomy_level": "supervised",
            "data_access": ["internal"],
            "external_apis": [],
            "target_audience": "Пользователи системы",
            "operations_per_hour": None,
            "revenue_per_operation": None,
            "system_prompts": [],
            "guardrails": [],
            "source_files": [],
            "detailed_summary": {
                "overview": f"Базовый обзор агента {preliminary_name}",
                "technical_architecture": "Техническая архитектура не определена",
                "operational_model": "Операционная модель не определена"
            }
        }

    async def _create_agent_profile_enhanced(self, llm_results: Dict[str, Any], parsed_data: Dict[str, Any],
                                             preliminary_name: str, assessment_id: str) -> AgentProfile:
        """Создание улучшенного профиля агента с лучшей обработкой ошибок"""

        bound_logger = self.logger.bind_context(assessment_id, self.name)
        bound_logger.info("🔄 Создание профиля агента...")

        try:
            # Формируем улучшенные данные для анализа
            enhanced_analysis_data = self._prepare_enhanced_analysis_data(llm_results, parsed_data)

            # Создаем продвинутый промпт для профилирования
            advanced_extraction_prompt = self._create_advanced_profile_prompt()

            bound_logger.info("🤖 Отправка запроса к LLM для создания профиля...")

            # Вызываем LLM для создания профиля
            llm_result = await self.call_llm_structured(
                data_to_analyze=enhanced_analysis_data,
                extraction_prompt=advanced_extraction_prompt,
                assessment_id=assessment_id,
                expected_format="JSON"
            )

            bound_logger.info("✅ Получен ответ от LLM")

            # ИСПРАВЛЕНО: Проверяем качество ответа LLM
            if not llm_result or not isinstance(llm_result, dict):
                bound_logger.warning("⚠️ LLM вернул некорректный ответ, создаем fallback профиль")
                llm_result = self._create_fallback_llm_result(preliminary_name)

            # Валидируем и дополняем результат
            profile_data = self._validate_and_enhance_profile_data(llm_result, preliminary_name, parsed_data)

            # Создаем объект AgentProfile
            agent_profile = self._construct_agent_profile(profile_data)

            bound_logger.info(f"✅ Профиль агента создан: {agent_profile.name} ({agent_profile.agent_type.value})")
            return agent_profile

        except Exception as e:
            bound_logger.error(f"❌ Ошибка создания профиля агента: {e}")

            # Создаем минимальный fallback профиль
            fallback_data = self._create_fallback_llm_result(preliminary_name)
            fallback_data = self._validate_and_enhance_profile_data(fallback_data, preliminary_name, parsed_data)

            return self._construct_agent_profile(fallback_data)

    def _prepare_enhanced_analysis_data(self, llm_results: Dict[str, Any], parsed_data: Dict[str, Any]) -> str:
        """Подготовка улучшенных данных для анализа"""
        data_sections = []

        # Добавляем результаты LLM анализа
        if llm_results:
            data_sections.append("=== РЕЗУЛЬТАТЫ КОНТЕКСТНОГО АНАЛИЗА ===")
            for context_type, context_result in llm_results.items():
                if isinstance(context_result, dict) and 'aggregated_analysis' in context_result:
                    analysis = context_result['aggregated_analysis']
                    data_sections.append(f"\n--- {context_type.upper().replace('_', ' ')} ---")
                    if isinstance(analysis, dict):
                        for key, value in analysis.items():
                            if isinstance(value, (str, list)) and value:
                                data_sections.append(f"{key}: {value}")
                    else:
                        data_sections.append(str(analysis))

        # Добавляем статистику парсинга
        if 'parsing_stats' in parsed_data:
            stats = parsed_data['parsing_stats']
            data_sections.append(f"\n=== СТАТИСТИКА ФАЙЛОВ ===")
            data_sections.append(f"Всего файлов: {stats['total_files']}")
            data_sections.append(f"Обработано: {stats['processed_files']}")
            data_sections.append(f"Ошибок: {stats['error_files']}")

        # Добавляем саммари кода
        if 'code_analysis' in parsed_data:
            code_summary = parsed_data['code_analysis']
            data_sections.append(f"\n=== АНАЛИЗ КОДА ===")
            data_sections.append(f"Языки программирования: {code_summary.get('languages', {})}")
            data_sections.append(f"Всего строк кода: {code_summary.get('total_lines', 0)}")
            data_sections.append(f"Индикаторы сложности: {code_summary.get('complexity_indicators', [])}")

        # Добавляем саммари промптов
        if 'prompt_analysis' in parsed_data:
            prompt_summary = parsed_data['prompt_analysis']
            data_sections.append(f"\n=== АНАЛИЗ ПРОМПТОВ ===")
            data_sections.append(f"Найдено промптов: {prompt_summary.get('total_prompts', 0)}")
            data_sections.append(f"Есть системные промпты: {prompt_summary.get('has_system_prompts', False)}")
            data_sections.append(f"Есть ограничения: {prompt_summary.get('has_guardrails', False)}")
            data_sections.append(f"Оценка сложности: {prompt_summary.get('complexity_score', 0)}")

        return "\n".join(data_sections)

    def _create_advanced_profile_prompt(self) -> str:
        """Создание продвинутого промпта для профилирования"""
        return json_profiler_extr_prompt

    def _enhance_with_parsed_data(self, profile_data: Dict[str, Any], parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Улучшение профиля данными из парсинга"""

        # Улучшение source_files
        file_types = set()
        if parsed_data.get("documents"):
            file_types.add("documents")
        if parsed_data.get("code_files"):
            file_types.add("code")
        if parsed_data.get("config_files"):
            file_types.add("configurations")
        if parsed_data.get("prompt_files"):
            file_types.add("prompts")

        profile_data["source_files"] = list(file_types)

        # Улучшение технических данных
        if 'code_analysis' in parsed_data:
            code_analysis = parsed_data['code_analysis']
            languages = list(code_analysis.get('languages', {}).keys())

            # Определяем тип агента на основе языков
            if not profile_data.get("agent_type") or profile_data["agent_type"] == "other":
                if "python" in languages:
                    profile_data["agent_type"] = "analyzer"
                elif "javascript" in languages:
                    profile_data["agent_type"] = "assistant"

        # Улучшение промптов
        if 'prompt_analysis' in parsed_data:
            prompt_analysis = parsed_data['prompt_analysis']
            if prompt_analysis.get('has_system_prompts') and not profile_data.get("system_prompts"):
                profile_data["system_prompts"] = ["Системные промпты найдены в коде/конфигурации"]

            if prompt_analysis.get('has_guardrails') and not profile_data.get("guardrails"):
                profile_data["guardrails"] = ["Ограничения найдены в промптах"]

        # ИСПРАВЛЕНО: Извлекаем LLM модель из parsed_data
        if 'code_analysis' in parsed_data and profile_data.get("llm_model") == "unknown":
            # Ищем упоминания LLM моделей в коде
            complexity_indicators = parsed_data['code_analysis'].get('complexity_indicators', [])
            if 'ai_ml_related' in complexity_indicators:
                profile_data["llm_model"] = "qwen3-4b"  # Можно улучшить определение модели

        return profile_data

    # 5. ИСПРАВЛЕНИЕ: Восстанавливаем отсутствующий метод _validate_enum_fields
    def _validate_enum_fields(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация полей с енумами"""

        # Валидация agent_type
        valid_agent_types = ["chatbot", "assistant", "trader", "scorer", "analyzer", "generator", "other"]
        if profile_data["agent_type"] not in valid_agent_types:
            profile_data["agent_type"] = "other"

        # Валидация autonomy_level
        valid_autonomy_levels = ["supervised", "semi_autonomous", "autonomous"]
        if profile_data["autonomy_level"] not in valid_autonomy_levels:
            profile_data["autonomy_level"] = "supervised"

        # Валидация data_access
        valid_data_sensitivities = ["public", "internal", "confidential", "critical"]
        validated_data_access = []
        for da in profile_data.get("data_access", []):
            if da in valid_data_sensitivities:
                validated_data_access.append(da)
        if not validated_data_access:
            validated_data_access = ["internal"]
        profile_data["data_access"] = validated_data_access

        return profile_data

    # 6. ИСПРАВЛЕНИЕ: Восстанавливаем отсутствующий метод _create_fallback_detailed_summary
    def _create_fallback_detailed_summary(self, profile_data: Dict[str, Any], parsed_data: Dict[str, Any]) -> Dict[
        str, str]:
        """Создание fallback детального саммари"""

        agent_name = profile_data.get("name", "Unknown Agent")
        agent_type = profile_data.get("agent_type", "other")

        # Собираем информацию из parsed_data
        file_count = sum([
            len(parsed_data.get("documents", {})),
            len(parsed_data.get("code_files", {})),
            len(parsed_data.get("config_files", {})),
            len(parsed_data.get("prompt_files", {}))
        ])

        languages = []
        if 'code_analysis' in parsed_data:
            languages = list(parsed_data['code_analysis'].get('languages', {}).keys())

        return {
            "overview": f"{agent_name} - это {agent_type} агент, проанализированный на основе {file_count} файлов. "
                        f"Агент предназначен для работы с пользователями и выполнения специализированных задач. "
                        f"Анализ показывает наличие структурированного кода и конфигураций.",

            "technical_architecture": f"Техническая архитектура агента основана на следующих компонентах: "
                                      f"языки программирования - {', '.join(languages) if languages else 'не определены'}, "
                                      f"файлы конфигурации, документация. "
                                      f"Агент использует LLM модель {profile_data.get('llm_model', 'unknown')} "
                                      f"с уровнем автономности {profile_data.get('autonomy_level', 'supervised')}. "
                                      f"Архитектура поддерживает обработку данных типа {', '.join(profile_data.get('data_access', ['internal']))}.",

            "operational_model": f"Операционная модель агента предполагает {profile_data.get('autonomy_level', 'supervised')} режим работы. "
                                 f"Агент взаимодействует с {profile_data.get('target_audience', 'пользователями')} "
                                 f"и обрабатывает запросы согласно заложенной логике. "
                                 f"{'Системные промпты определяют поведение агента.' if profile_data.get('system_prompts') else 'Системные промпты не обнаружены.'} "
                                 f"{'Встроенные ограничения обеспечивают безопасность.' if profile_data.get('guardrails') else 'Ограничения безопасности не определены.'}"
        }


    def _validate_and_enhance_profile_data(self, llm_result: Dict[str, Any], preliminary_name: str,
                                           parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация и улучшение данных профиля"""

        # ИСПРАВЛЕНО: Более детальная валидация с логированием
        self.logger.bind_context("unknown", self.name).info("🔍 Валидация данных профиля от LLM...")

        # Базовая валидация с умными дефолтами
        defaults = {
            "name": preliminary_name,
            "version": "1.0",
            "description": "ИИ-агент (описание сгенерировано автоматически)",
            "agent_type": "other",
            "llm_model": "unknown",
            "autonomy_level": "supervised",
            "data_access": ["internal"],
            "external_apis": [],
            "target_audience": "Пользователи системы",
            "operations_per_hour": None,
            "revenue_per_operation": None,
            "system_prompts": [],
            "guardrails": [],
            "source_files": []
        }

        # ИСПРАВЛЕНО: Проверяем что LLM вернул валидный результат
        if not isinstance(llm_result, dict):
            self.logger.bind_context("unknown", self.name).warning("⚠️ LLM вернул не-словарь, используем defaults")
            llm_result = {}

        # Применяем дефолты
        for key, default_value in defaults.items():
            if key not in llm_result or llm_result[key] is None or llm_result[key] == "":
                llm_result[key] = default_value
                self.logger.bind_context("unknown", self.name).debug(
                    f"🔧 Поле '{key}' заменено на default: {default_value}")

        # ИСПРАВЛЕНО: Улучшаем данные на основе parsed_data
        llm_result = self._enhance_with_parsed_data(llm_result, parsed_data)

        # ИСПРАВЛЕНО: Валидация енумов
        llm_result = self._validate_enum_fields(llm_result)

        # ИСПРАВЛЕНО: Обеспечиваем detailed_summary
        if 'detailed_summary' not in llm_result or not llm_result['detailed_summary']:
            llm_result['detailed_summary'] = self._create_fallback_detailed_summary(llm_result, parsed_data)

        self.logger.bind_context("unknown", self.name).info("✅ Валидация данных профиля завершена")
        return llm_result

    def _create_agent_profile(self, profile_data: Dict[str, Any]) -> AgentProfile:
            """ИСПРАВЛЕННОЕ создание объекта AgentProfile"""

            try:
                # Преобразуем строковые значения в enum'ы
                agent_type = AgentType(profile_data.get("agent_type", "other"))
                autonomy_level = AutonomyLevel(profile_data.get("autonomy_level", "supervised"))

                # Преобразуем data_access в enum'ы
                data_access_list = []
                for da in profile_data.get("data_access", ["internal"]):
                    try:
                        data_access_list.append(DataSensitivity(da))
                    except ValueError:
                        data_access_list.append(DataSensitivity.INTERNAL)

                return AgentProfile(
                    name=profile_data.get("name", "Unknown Agent"),
                    version=profile_data.get("version", "1.0"),
                    description=profile_data.get("description", "Автоматически сгенерированное описание"),
                    agent_type=agent_type,
                    llm_model=profile_data.get("llm_model", "unknown"),
                    autonomy_level=autonomy_level,
                    data_access=data_access_list,
                    external_apis=profile_data.get("external_apis", []),
                    target_audience=profile_data.get("target_audience", "Общая аудитория"),
                    operations_per_hour=profile_data.get("operations_per_hour"),
                    revenue_per_operation=profile_data.get("revenue_per_operation"),
                    system_prompts=profile_data.get("system_prompts", []),
                    guardrails=profile_data.get("guardrails", []),
                    source_files=profile_data.get("source_files", []),
                    detailed_summary=profile_data.get("detailed_summary"),
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )

            except Exception as e:
                self.logger.bind_context("unknown", self.name).error(
                    f"Ошибка создания AgentProfile: {e}"
                )

                # Fallback профиль
                return AgentProfile(
                    name=profile_data.get("name", "Unknown Agent"),
                    version="1.0",
                    description="Fallback профиль из-за ошибки создания",
                    agent_type=AgentType.OTHER,
                    llm_model="unknown",
                    autonomy_level=AutonomyLevel.SUPERVISED,
                    data_access=[DataSensitivity.INTERNAL],
                    external_apis=[],
                    target_audience="Общая аудитория",
                    system_prompts=[],
                    guardrails=[],
                    source_files=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                    )

    def _serialize_agent_profile_for_result(self, agent_profile: AgentProfile) -> Dict[str, Any]:
        """Сериализация AgentProfile для результата"""
        return {
            "name": agent_profile.name,
            "version": agent_profile.version,
            "description": agent_profile.description,
            "agent_type": agent_profile.agent_type.value,
            "llm_model": agent_profile.llm_model,
            "autonomy_level": agent_profile.autonomy_level.value,
            "data_access": [ds.value for ds in agent_profile.data_access],
            "external_apis": agent_profile.external_apis,
            "target_audience": agent_profile.target_audience,
            "operations_per_hour": agent_profile.operations_per_hour,
            "revenue_per_operation": agent_profile.revenue_per_operation,
            "system_prompts": agent_profile.system_prompts,
            "guardrails": agent_profile.guardrails,
            "source_files": agent_profile.source_files,
            "detailed_summary": agent_profile.detailed_summary,
            "created_at": agent_profile.created_at.isoformat() if agent_profile.created_at else None,
            "updated_at": agent_profile.updated_at.isoformat() if agent_profile.updated_at else None
        }

    async def _save_outputs(self, outputs: Dict[str, str], assessment_id: str) -> List[str]:
        """Сохранение выходных файлов"""
        saved_files = []

        try:
                # Создаем директорию для результатов
            output_dir = Path(f"outputs/{assessment_id}")
            output_dir.mkdir(parents=True, exist_ok=True)

                # Сохраняем каждый тип выхода
            for output_type, content in outputs.items():
                if not content:
                    continue

                if output_type == "summary_report":
                    file_path = output_dir / "summary_report.md"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    saved_files.append(str(file_path))

                elif output_type == "architecture_graph":
                    file_path = output_dir / "architecture.mermaid"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    saved_files.append(str(file_path))

                elif output_type == "detailed_json":
                    file_path = output_dir / "detailed_analysis.json"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    saved_files.append(str(file_path))

                elif output_type == "processing_log":
                    file_path = output_dir / "processing_log.txt"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    saved_files.append(str(file_path))

            self.logger.bind_context(assessment_id, self.name).info(
                f"📁 Сохранено {len(saved_files)} файлов результатов"
            )

        except Exception as e:
            self.logger.bind_context(assessment_id, self.name).error(
                f"Ошибка сохранения результатов: {e}"
            )

        return saved_files

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Расчет метрик производительности"""
        total_time = sum(
            (stage.end_time - stage.start_time).total_seconds()
            for stage in self.processing_stages
            if stage.end_time
        )

        return {
            "total_stages": len(self.processing_stages),
            "successful_stages": len([s for s in self.processing_stages if s.status == "completed"]),
            "failed_stages": len([s for s in self.processing_stages if s.status == "failed"]),
            "total_processing_time": total_time,
            "avg_stage_time": total_time / len(self.processing_stages) if self.processing_stages else 0,
            "cache_hits": len(self.llm_orchestrator.processing_cache),
            "chunks_processed": sum(
                stage.metrics.get("chunks_created", 0)
                for stage in self.processing_stages
                if stage.metrics
            )
        }

    # ===============================
    # Фабричные функции для создания профайлера


def create_profiler_from_env() -> EnhancedProfilerAgent:
    """Создание профайлера из переменных окружения"""
    from .base_agent import create_default_config_from_env

    config = create_default_config_from_env()
    config.name = "enhanced_profiler"
    config.description = "Улучшенный профайлер ИИ-агентов с новой архитектурой"

    return EnhancedProfilerAgent(config)


def create_profiler_node_function(profiler: EnhancedProfilerAgent):
    """Создание функции узла для LangGraph"""

    async def profiler_node(state: WorkflowState) -> WorkflowState:
        """Узел профилирования для LangGraph"""

        try:
            assessment_id = state.get("assessment_id", "unknown")
            source_files = state.get("source_files", [])
            agent_name = state.get("preliminary_agent_name", "Unknown Agent")

            if not source_files:
                state.update({
                    "current_step": "error",
                    "error_message": "Не предоставлены файлы для анализа"
                })
                return state

            input_data = {
                "source_files": source_files,
                "agent_name": agent_name
            }

            # Запускаем профилирование
            result = await profiler.process(input_data, assessment_id)

            if result.status == ProcessingStatus.COMPLETED:
                # ИСПРАВЛЕНО: Правильное сохранение в состояние
                state.update({
                    "agent_profile": result.result_data.get("agent_profile", {}),
                    "profiling_result": result.result_data,  # Сохраняем ВСЕ данные профилирования
                    "current_step": "evaluation_preparation"
                })

                print(f"🔍 DEBUG profiler_node: Сохранили в состояние:")
                print(f"  - agent_profile: {bool(result.result_data.get('agent_profile'))}")
                print(f"  - llm_analysis_results: {bool(result.result_data.get('llm_analysis_results'))}")
                print(f"  - output_files: {len(result.result_data.get('output_files', []))}")

            else:
                state.update({
                    "current_step": "error",
                    "error_message": result.error_message or "Ошибка профилирования"
                })

            return state

        except Exception as e:
            state.update({
                "current_step": "error",
                "error_message": f"Исключение в профайлере: {str(e)}"
            })
            return state

    return profiler_node
    # ===============================
    # Экспорт
    # ===============================

__all__ = [
        "EnhancedProfilerAgent",
        "FileSystemCrawler",
        "ContextAwareChunker",
        "LLMOrchestrator",
        "OutputGenerator",
        "FileMetadata",
        "ContextChunk",
        "ProcessingStage",
        "create_profiler_from_env",
        "create_profiler_node_function"
    ]