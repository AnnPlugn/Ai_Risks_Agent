# src/agents/profiler_agent.py
"""
Профайлер-агент для сбора и анализа данных об ИИ-агенте
Собирает информацию из кодовой базы, документации, промптов и конфигураций
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .base_agent import AnalysisAgent, AgentConfig
from ..models.risk_models import AgentProfile, AgentTaskResult, ProcessingStatus, AgentType, AutonomyLevel, DataSensitivity
from ..tools.document_parser import create_document_parser, parse_agent_documents
from ..tools.code_analyzer import create_code_analyzer, analyze_agent_codebase
from ..tools.prompt_analyzer import create_prompt_analyzer, analyze_agent_prompts
from ..utils.logger import LogContext


class ProfilerAgent(AnalysisAgent):
    """
    Агент-профайлер для сбора данных об ИИ-агенте
    
    Функции:
    1. Парсинг документации (Word, Excel, PDF)
    2. Анализ кодовой базы (Python, JavaScript, Java)
    3. Извлечение промптов и инструкций
    4. Определение технических характеристик
    5. Создание профиля агента для оценки рисков
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Инициализируем инструменты анализа
        self.document_parser = create_document_parser()
        self.code_analyzer = create_code_analyzer()
        self.prompt_analyzer = create_prompt_analyzer()
    
    def get_system_prompt(self) -> str:
        """Системный промпт для профайлера"""
        return """Ты - эксперт-аналитик по профилированию ИИ-агентов для оценки операционных рисков.

Твоя задача: создавать детальные профили ИИ-агентов на основе доступной информации.

ПРИНЦИПЫ АНАЛИЗА:
1. Тщательно анализируй все предоставленные данные
2. Определяй назначение и возможности агента
3. Выявляй технические характеристики
4. Оценивай уровень автономности
5. Классифицируй доступ к данным
6. Извлекай ключевые промпты и ограничения

ФОРМАТ ВЫВОДА: Структурированный JSON с полным профилем агента

ОБЯЗАТЕЛЬНЫЕ ПОЛЯ:
- name: название агента
- agent_type: тип агента (chatbot, assistant, trader, scorer, analyzer, generator)
- autonomy_level: уровень автономности (supervised, semi_autonomous, autonomous)
- data_access: типы данных (public, internal, confidential, critical)
- target_audience: целевая аудитория
- llm_model: используемая модель
- system_prompts: системные промпты
- guardrails: ограничения безопасности

Будь точным и объективным в анализе."""
    
    async def process(
        self, 
        input_data: Dict[str, Any], 
        assessment_id: str
    ) -> AgentTaskResult:
        """
        Основная обработка профайлинга агента
        
        Args:
            input_data: Содержит пути к файлам и папкам для анализа
                - source_files: List[str] - список файлов/папок
                - agent_name: Optional[str] - предварительное имя агента
            assessment_id: ID оценки
            
        Returns:
            Результат с профилем агента
        """
        start_time = datetime.now()
        
        try:
            with LogContext("profile_agent", assessment_id, self.name):
                # Извлекаем входные данные
                source_files = input_data.get("source_files", [])
                preliminary_name = input_data.get("agent_name", "Unknown_Agent")
                
                if not source_files:
                    raise ValueError("Не предоставлены файлы для анализа")
                
                # Собираем данные из всех источников
                collected_data = await self._collect_all_data(source_files, assessment_id)
                
                # Анализируем собранные данные с помощью LLM
                agent_profile = await self._analyze_and_create_profile(
                    collected_data, preliminary_name, assessment_id
                )
                
                # Создаем результат
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                return AgentTaskResult(
                    agent_name=self.name,
                    task_type="profiling",
                    status=ProcessingStatus.COMPLETED,
                    result_data={
                        "agent_profile": agent_profile.dict(),
                        
                        "collected_data_summary": self._create_data_summary(collected_data)
                        
                    },
                    start_time=start_time,
                    end_time=end_time,
                    execution_time_seconds=execution_time
                )
                
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return AgentTaskResult(
                agent_name=self.name,
                task_type="profiling",
                status=ProcessingStatus.FAILED,
                error_message=str(e),
                start_time=start_time,
                end_time=end_time,
                execution_time_seconds=execution_time
            )
    
    async def _collect_all_data(
        self, 
        source_files: List[str], 
        assessment_id: str
    ) -> Dict[str, Any]:
        """Сбор данных из всех источников"""
        
        collected_data = {
            "documents": [],
            "code_analysis": None,
            "prompt_analysis": None,
            "source_files": source_files,
            "errors": []
        }
        
        # Разделяем файлы и папки
        files_to_parse = []
        directories_to_analyze = []
        
        for source in source_files:
            path = Path(source)
            if path.is_file():
                files_to_parse.append(path)
            elif path.is_dir():
                directories_to_analyze.append(path)
                # Добавляем файлы из директории
                for file_path in path.rglob('*'):
                    if file_path.is_file():
                        files_to_parse.append(file_path)
        
        # 1. Парсинг документов
        self.logger.bind_context(assessment_id, self.name).info(
            f"📄 Парсинг {len(files_to_parse)} файлов"
        )
        
        try:
            parsed_docs, agent_info = parse_agent_documents(
                files_to_parse, 
                extract_agent_info=True
            )
            
            collected_data["documents"] = [
                {
                    "file_path": doc.file_path,
                    "file_type": doc.file_type,
                    "content": doc.content,
                    "sections": doc.sections,
                    "tables": doc.tables,
                    "success": doc.success
                }
                for doc in parsed_docs
            ]
            
            collected_data["document_agent_info"] = agent_info
            
        except Exception as e:
            collected_data["errors"].append(f"Ошибка парсинга документов: {e}")
        
        # 2. Анализ кодовой базы
        self.logger.bind_context(assessment_id, self.name).info(
            f"💻 Анализ кода в {len(directories_to_analyze)} директориях"
        )
        
        for directory in directories_to_analyze:
            try:
                code_analysis = analyze_agent_codebase(directory, max_files=50)
                
                if code_analysis.success:
                    collected_data["code_analysis"] = {
                        "project_path": code_analysis.project_path,
                        "total_files": code_analysis.total_files,
                        "total_lines": code_analysis.total_lines,
                        "languages": code_analysis.languages,
                        "dependencies": code_analysis.dependencies,
                        "entry_points": code_analysis.entry_points,
                        "security_summary": code_analysis.security_summary,
                        "complexity_summary": code_analysis.complexity_summary
                    }
                    break  # Берем первую успешную директорию
                
            except Exception as e:
                collected_data["errors"].append(f"Ошибка анализа кода {directory}: {e}")
        
        # 3. Анализ промптов
        self.logger.bind_context(assessment_id, self.name).info(
            "🔍 Анализ промптов и инструкций"
        )
        
        try:
            # Извлекаем тексты для анализа промптов
            prompt_sources = []
            
            # Из документов
            for doc in collected_data["documents"]:
                if doc["success"]:
                    # Добавляем содержимое секций с промптами
                    for section_name, section_content in doc["sections"].items():
                        if any(keyword in section_name.lower() for keyword in 
                               ['prompt', 'instruction', 'system', 'guardrail']):
                            prompt_sources.append(section_content)
                    
                    # Добавляем общий контент если он небольшой
                    if len(doc["content"]) < 5000:
                        prompt_sources.append(doc["content"])
            
            # Из кода (комментарии и строки)
            if collected_data["code_analysis"]:
                # Анализируем сами файлы кода
                for file_path in files_to_parse:
                    if file_path.suffix.lower() in ['.py', '.js', '.java']:
                        prompt_sources.append(str(file_path))
            
            if prompt_sources:
                prompt_analysis = analyze_agent_prompts(prompt_sources)
                
                if prompt_analysis.success:
                    collected_data["prompt_analysis"] = {
                        "total_prompts": prompt_analysis.total_prompts,
                        "system_prompts": [p.content for p in prompt_analysis.system_prompts],
                        "guardrails": [p.content for p in prompt_analysis.guardrails],
                        "capabilities": prompt_analysis.capabilities,
                        "personality_traits": prompt_analysis.personality_traits,
                        "restrictions": prompt_analysis.restrictions,
                        "risk_indicators": prompt_analysis.risk_indicators,
                        "complexity_score": prompt_analysis.complexity_score
                    }
                
        except Exception as e:
            collected_data["errors"].append(f"Ошибка анализа промптов: {e}")
        
        return collected_data
    
    async def _analyze_and_create_profile(
        self,
        collected_data: Dict[str, Any],
        preliminary_name: str,
        assessment_id: str
    ) -> AgentProfile:
        """Анализ собранных данных и создание профиля агента"""
        
        # Формируем данные для анализа LLM
        analysis_data = self._format_data_for_llm(collected_data)
        
        # Создаем промпт для извлечения профиля
        extraction_prompt = """/no_think
        Проанализируй предоставленные данные об ИИ-агенте и создай детальный JSON профиль.
        
        
ОБЯЗАТЕЛЬНЫЕ ПОЛЯ В JSON:
{
    "name": "string - название агента",
    "version": "string - версия (по умолчанию 1.0)",
    "description": "string - описание назначения агента",
    "agent_type": "string - один из: chatbot, assistant, trader, scorer, analyzer, generator, other",
    "llm_model": "string - используемая LLM модель",
    "autonomy_level": "string - один из: supervised, semi_autonomous, autonomous",
    "data_access": ["array of strings - типы данных: public, internal, confidential, critical"],
    "external_apis": ["array of strings - внешние API"],
    "target_audience": "string - целевая аудитория",
    "operations_per_hour": "number or null - операций в час",
    "revenue_per_operation": "number or null - доход с операции в рублях",
    "system_prompts": ["array of strings - системные промпты"],
    "guardrails": ["array of strings - ограничения безопасности"],
    "technical_details": "дополнительные технические подробности о реализации",
    "risk_factors": ["выявленные факторы риска"],
    "capabilities": ["полный список возможностей агента"],
    "limitations": ["известные ограничения"],
    "detailed_summary": {
            "overview": "Подробное описание назначения, функций и области применения агента - минимум 200 слов",
            "technical_architecture": "Детальный анализ технической реализации, используемых технологий, интеграций - минимум 150 слов",
            "operational_model": "Описание процессов работы, взаимодействия с пользователями, бизнес-логики - минимум 150 слов",
            "risk_analysis": "Выявленные риски, уязвимости, потенциальные проблемы с детальным обоснованием - минимум 200 слов",
            "security_recommendations": "Конкретные меры по снижению рисков и улучшению безопасности - минимум 100 слов",
            "conclusions": "Итоговая оценка агента, ключевые моменты для внимания - минимум 100 слов"
        }
}
Создай максимально подробный профиль с детальным анализом всех аспектов агента.
=== ДЕТАЛЬНОЕ САММАРИ ===

ОБЗОР АГЕНТА:
[Подробное описание назначения, функций и области применения агента - минимум 200 слов]

ТЕХНИЧЕСКАЯ АРХИТЕКТУРА:
[Детальный анализ технической реализации, используемых технологий, интеграций - минимум 150 слов]

ОПЕРАЦИОННАЯ МОДЕЛЬ:
[Описание процессов работы, взаимодействия с пользователями, бизнес-логики - минимум 150 слов]

АНАЛИЗ РИСКОВ:
[Выявленные риски, уязвимости, потенциальные проблемы с детальным обоснованием - минимум 200 слов]

РЕКОМЕНДАЦИИ ПО БЕЗОПАСНОСТИ:
[Конкретные меры по снижению рисков и улучшению безопасности - минимум 100 слов]

ВЫВОДЫ:
[Итоговая оценка агента, ключевые моменты для внимания - минимум 100 слов]



"""
#Вернуть только JSON без объяснений, комментариев или других текстов.
        
        # Вызываем LLM для создания профиля
        llm_result = await self.call_llm_structured(
            data_to_analyze=analysis_data,
            extraction_prompt=extraction_prompt,
            assessment_id=assessment_id,
            expected_format="JSON"
        )
        
        # Создаем объект AgentProfile
        profile_data = self._validate_and_fix_profile_data(llm_result, preliminary_name)
        
        agent_profile = AgentProfile(
            name=profile_data["name"],
            version=profile_data.get("version", "1.0"),
            description=profile_data["description"],
            agent_type=AgentType(profile_data["agent_type"]),
            llm_model=profile_data["llm_model"],
            autonomy_level=AutonomyLevel(profile_data["autonomy_level"]),
            data_access=[DataSensitivity(da) for da in profile_data.get("data_access", [])],
            external_apis=profile_data.get("external_apis", []),
            target_audience=profile_data["target_audience"],
            operations_per_hour=profile_data.get("operations_per_hour"),
            revenue_per_operation=profile_data.get("revenue_per_operation"),
            system_prompts=profile_data.get("system_prompts", []),
            guardrails=profile_data.get("guardrails", []),
            source_files=collected_data.get("source_files", []),
            detailed_summary=profile_data.get("detailed_summary")
        )
        
        if hasattr(agent_profile, 'detailed_summary') and agent_profile.detailed_summary:
            # Логируем успешное создание саммари
            summary_sections = list(agent_profile.detailed_summary.keys())
            self.logger.bind_context(assessment_id, self.name).info(
                f"✅ Профайлер создал детальное саммари с разделами: {', '.join(summary_sections)}"
            )
            
            # Дополнительная диагностика размеров
            total_summary_length = sum(len(str(section)) for section in agent_profile.detailed_summary.values())
            self.logger.bind_context(assessment_id, self.name).debug(
                f"📊 Общий размер детального саммари: {total_summary_length} символов"
            )
            
            # Проверяем каждый раздел
            for section_name, section_content in agent_profile.detailed_summary.items():
                section_length = len(str(section_content))
                if section_length < 50:
                    self.logger.bind_context(assessment_id, self.name).warning(
                        f"⚠️ Раздел '{section_name}' слишком короткий: {section_length} символов"
                    )
                else:
                    self.logger.bind_context(assessment_id, self.name).debug(
                        f"✅ Раздел '{section_name}': {section_length} символов"
                    )
        else:
            # Логируем отсутствие саммари
            self.logger.bind_context(assessment_id, self.name).warning(
                "⚠️ Детальное саммари НЕ создано профайлером"
            )
            
            # Диагностируем причины
            if not hasattr(agent_profile, 'detailed_summary'):
                self.logger.bind_context(assessment_id, self.name).error(
                    "❌ У AgentProfile отсутствует поле detailed_summary - проверьте модель данных"
                )
            elif agent_profile.detailed_summary is None:
                self.logger.bind_context(assessment_id, self.name).warning(
                    "⚠️ Поле detailed_summary равно None - проверьте промпт и ответ LLM"
                )
        
        return agent_profile

    def _format_data_for_llm(self, collected_data: Dict[str, Any]) -> str:
        """Форматирование собранных данных для анализа LLM и сохранение в файл"""

        formatted_parts = []

        # 1. Информация из документов
        if collected_data.get("documents"):
            formatted_parts.append("=== ДОКУМЕНТАЦИЯ ===")

            for doc in collected_data["documents"]:
                if doc["success"]:
                    formatted_parts.append(f"\nФайл: {Path(doc['file_path']).name}")
                    formatted_parts.append(f"Тип: {doc['file_type']}")

                    # Секции документа
                    for section_name, section_content in doc["sections"].items():
                        if section_content.strip():
                            formatted_parts.append(f"\n[{section_name.upper()}]")
                            # Ограничиваем длину секции
                            content = section_content[:1000] + "..." if len(section_content) > 1000 else section_content
                            formatted_parts.append(content)

                    # Таблицы
                    if doc["tables"]:
                        formatted_parts.append(f"\nТаблиц найдено: {len(doc['tables'])}")

        # 2. Анализ кода
        if collected_data.get("code_analysis"):
            code_data = collected_data["code_analysis"]
            formatted_parts.append("\n\n=== АНАЛИЗ КОДА ===")
            formatted_parts.append(f"Проект: {code_data['project_path']}")
            formatted_parts.append(f"Файлов: {code_data['total_files']}")
            formatted_parts.append(f"Строк кода: {code_data['total_lines']}")
            formatted_parts.append(f"Языки: {', '.join(code_data['languages'].keys())}")

            if code_data.get("dependencies"):
                formatted_parts.append(f"\nЗависимости:")
                dep_count = 0
                for file_path, deps in code_data["dependencies"].items():
                    if dep_count < 20:  # Ограничиваем количество
                        formatted_parts.append(f"  {Path(file_path).name}: {', '.join(deps[:5])}")
                        dep_count += len(deps)

            if code_data.get("entry_points"):
                formatted_parts.append(f"\nТочки входа: {', '.join(code_data['entry_points'])}")

            # Безопасность и сложность
            security = code_data.get("security_summary", {})
            complexity = code_data.get("complexity_summary", {})

            formatted_parts.append(f"\nБезопасность: {security.get('total_issues', 0)} проблем")
            formatted_parts.append(f"Средняя сложность: {complexity.get('average_complexity', 0):.1f}")

        # 3. Анализ промптов
        if collected_data.get("prompt_analysis"):
            prompt_data = collected_data["prompt_analysis"]
            formatted_parts.append("\n\n=== АНАЛИЗ ПРОМПТОВ ===")
            formatted_parts.append(f"Всего промптов: {prompt_data['total_prompts']}")

            if prompt_data.get("system_prompts"):
                formatted_parts.append(f"\nСистемные промпты:")
                for i, prompt in enumerate(prompt_data["system_prompts"][:3]):  # Первые 3
                    formatted_parts.append(f"  {i + 1}. {prompt[:200]}...")

            if prompt_data.get("guardrails"):
                formatted_parts.append(f"\nОграничения:")
                for i, guardrail in enumerate(prompt_data["guardrails"][:3]):
                    formatted_parts.append(f"  {i + 1}. {guardrail[:200]}...")

            if prompt_data.get("capabilities"):
                formatted_parts.append(f"\nВозможности: {', '.join(prompt_data['capabilities'])}")

            if prompt_data.get("risk_indicators"):
                formatted_parts.append(f"Индикаторы риска: {', '.join(prompt_data['risk_indicators'])}")

        # 4. Ошибки (если есть)
        if collected_data.get("errors"):
            formatted_parts.append(f"\n\n=== ОШИБКИ СБОРА ДАННЫХ ===")
            for error in collected_data["errors"]:
                formatted_parts.append(f"- {error}")

        formatted_output = "\n".join(formatted_parts)

        # Сохранение в файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"llm_data_{timestamp}.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
        except Exception as e:
            formatted_parts.append(f"\n\n=== ОШИБКА ПРИ СОХРАНЕНИИ ===")
            formatted_parts.append(f"- Не удалось сохранить файл: {str(e)}")
            formatted_output = "\n".join(formatted_parts)

        return formatted_output
    
    def _validate_and_fix_profile_data(
        self, 
        llm_result: Dict[str, Any], 
        preliminary_name: str
    ) -> Dict[str, Any]:
        """Валидация и исправление данных профиля от LLM"""
        
        # Значения по умолчанию
        defaults = {
            "name": preliminary_name,
            "version": "1.0",
            "description": "ИИ-агент (описание не найдено)",
            "agent_type": "other",
            "llm_model": "unknown",
            "autonomy_level": "supervised",
            "data_access": ["internal"],
            "external_apis": [],
            "target_audience": "Пользователи системы",
            "operations_per_hour": None,
            "revenue_per_operation": None,
            "system_prompts": [],
            "guardrails": []
        }
        
        # Применяем значения по умолчанию
        for key, default_value in defaults.items():
            if key not in llm_result or llm_result[key] is None:
                llm_result[key] = default_value
        
        # Валидация енумов
        valid_agent_types = [e.value for e in AgentType]
        if llm_result["agent_type"] not in valid_agent_types:
            llm_result["agent_type"] = "other"
        
        valid_autonomy_levels = [e.value for e in AutonomyLevel]
        if llm_result["autonomy_level"] not in valid_autonomy_levels:
            llm_result["autonomy_level"] = "supervised"
        
        valid_data_sensitivities = [e.value for e in DataSensitivity]
        validated_data_access = []
        for da in llm_result.get("data_access", []):
            if da in valid_data_sensitivities:
                validated_data_access.append(da)
        if not validated_data_access:
            validated_data_access = ["internal"]
        llm_result["data_access"] = validated_data_access
        
        # Валидация списков
        for list_field in ["external_apis", "system_prompts", "guardrails"]:
            if not isinstance(llm_result.get(list_field), list):
                llm_result[list_field] = []
        
        return llm_result
    
    def _create_data_summary(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Создание краткой сводки собранных данных"""
        
        summary = {
            "documents_processed": 0,
            "documents_successful": 0,
            "code_analysis_success": False,
            "prompt_analysis_success": False,
            "total_source_files": len(collected_data.get("source_files", [])),
            "errors_count": len(collected_data.get("errors", []))
        }
        
        # Статистика документов
        if collected_data.get("documents"):
            summary["documents_processed"] = len(collected_data["documents"])
            summary["documents_successful"] = sum(
                1 for doc in collected_data["documents"] if doc["success"]
            )
        
        # Статистика анализа кода
        if collected_data.get("code_analysis"):
            summary["code_analysis_success"] = True
            summary["total_code_files"] = collected_data["code_analysis"]["total_files"]
            summary["total_code_lines"] = collected_data["code_analysis"]["total_lines"]
            summary["programming_languages"] = list(collected_data["code_analysis"]["languages"].keys())
        
        # Статистика анализа промптов
        if collected_data.get("prompt_analysis"):
            summary["prompt_analysis_success"] = True
            summary["total_prompts_found"] = collected_data["prompt_analysis"]["total_prompts"]
            summary["system_prompts_found"] = len(collected_data["prompt_analysis"]["system_prompts"])
            summary["guardrails_found"] = len(collected_data["prompt_analysis"]["guardrails"])
        
        return summary
    
    def _get_required_result_fields(self) -> List[str]:
        """Обязательные поля результата профайлера"""
        return ["agent_profile", "collected_data_summary"]


# ===============================
# Интеграция с LangGraph
# ===============================

def create_profiler_node_function(profiler_agent: ProfilerAgent):
    """
    Создает функцию узла для LangGraph workflow
    
    Args:
        profiler_agent: Экземпляр профайлер-агента
        
    Returns:
        Функция для использования в LangGraph
    """
    async def profiler_node(state: Dict[str, Any]) -> Dict[str, Any]:
        """Узел профайлера в LangGraph workflow"""
        
        # Извлекаем данные из состояния
        assessment_id = state.get("assessment_id", "unknown")
        source_files = state.get("source_files", [])
        agent_name = state.get("preliminary_agent_name", "Unknown_Agent")
        
        # Подготавливаем входные данные
        input_data = {
            "source_files": source_files,
            "agent_name": agent_name
        }
        
        # Запускаем профайлер
        result = await profiler_agent.run(input_data, assessment_id)
        
        # Обновляем состояние - преобразуем AgentTaskResult в словарь
        updated_state = state.copy()
        updated_state["profiling_result"] = result.dict()  # Преобразуем в словарь
        
        if result.status == ProcessingStatus.COMPLETED:
            # Добавляем профиль агента в состояние для дальнейшего использования
            agent_profile_data = result.result_data["agent_profile"]
            updated_state["agent_profile"] = agent_profile_data
            updated_state["current_step"] = "evaluation_preparation"
        else:
            updated_state["current_step"] = "error"
            updated_state["error_message"] = result.error_message
        
        return updated_state
    
    return profiler_node


# ===============================
# Фабрики
# ===============================

def create_profiler_agent(
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    temperature: Optional[float] = None
) -> ProfilerAgent:
    """
    Создание профайлер-агента
    ОБНОВЛЕНО: Использует центральный конфигуратор
    """
    from .base_agent import create_agent_config
    
    config = create_agent_config(
        name="profiler_agent",
        description="Агент для профилирования ИИ-агентов и сбора данных для оценки рисков",
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        temperature=temperature,
        max_retries=3,
        timeout_seconds=1800,  # Увеличенный тайм-аут для анализа больших объемов данных
        use_risk_analysis_client=False  # Профайлер использует стандартный клиент
    )
    
    return ProfilerAgent(config)

def create_profiler_from_env() -> ProfilerAgent:
    """
    Создание профайлер-агента из переменных окружения
    ОБНОВЛЕНО: Использует центральный конфигуратор
    """
    # ИЗМЕНЕНО: Используем центральный конфигуратор, убираем дублирование чтения env
    return create_profiler_agent()


# Экспорт
__all__ = [
    "ProfilerAgent",
    "create_profiler_agent",
    "create_profiler_from_env",
    "create_profiler_node_function"
]