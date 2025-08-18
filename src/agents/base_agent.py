# src/agents/base_agent.py
"""
Базовый класс для всех агентов в системе оценки рисков ИИ-агентов
Предоставляет общий интерфейс и функциональность для всех типов агентов
"""

"""
Базовый класс для всех агентов в системе оценки рисков ИИ-агентов
ИСПРАВЛЕНО: Правильная работа с DeepSeek клиентами
"""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from dataclasses import dataclass

from ..utils.llm_client import LLMClient, LLMConfig, LLMMessage, DeepSeekRiskAnalysisLLMClient, \
    GigaChatRiskAnalysisLLMClient, RiskAnalysisLLMClient
from ..utils.llm_config_manager import get_llm_config_manager, LLMProvider
from ..utils.logger import get_logger, log_agent_execution, log_llm_call
from ..models.risk_models import AgentTaskResult, ProcessingStatus


@dataclass
class AgentConfig:
    """Конфигурация агента"""
    name: str
    description: str
    llm_config: LLMConfig
    max_retries: int = 3
    timeout_seconds: int = 180
    temperature: float = 0.1
    use_risk_analysis_client: bool = False


class BaseAgent(ABC):
    """Базовый класс для всех агентов системы оценки рисков"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = get_logger()

        # ИСПРАВЛЕНО: Правильная фабрика для DeepSeek
        self.llm_client = self._create_appropriate_llm_client(config)

        # Статистика агента
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_execution_time": 0.0,
            "average_response_time": 0.0
        }

    def _create_appropriate_llm_client(self, config: AgentConfig):
        """НОВЫЙ МЕТОД: Создает правильный LLM клиент для текущего провайдера"""

        # Получаем актуальную информацию о провайдере
        manager = get_llm_config_manager()
        provider = manager.get_provider()

        print(f"🔍 DEBUG BaseAgent: Создаем клиент для провайдера {provider.value}")
        print(f"🔍 DEBUG BaseAgent: Нужен risk_analysis_client: {config.use_risk_analysis_client}")

        if config.use_risk_analysis_client:
            # Создаем специализированный клиент для анализа рисков
            if provider == LLMProvider.DEEPSEEK:
                print("✅ Создаем DeepSeekRiskAnalysisLLMClient")
                return DeepSeekRiskAnalysisLLMClient(config.llm_config)
            elif provider == LLMProvider.GIGACHAT:
                print("✅ Создаем GigaChatRiskAnalysisLLMClient")
                return GigaChatRiskAnalysisLLMClient(config.llm_config)
            else:
                print("✅ Создаем RiskAnalysisLLMClient (fallback)")
                return RiskAnalysisLLMClient(config.llm_config)
        else:
            # Создаем стандартный клиент
            from ..utils.llm_client import create_llm_client
            return create_llm_client(
                client_type="standard",
                base_url=config.llm_config.base_url,
                model=config.llm_config.model,
                temperature=config.llm_config.temperature
            )

    @property
    def name(self) -> str:
        """Имя агента"""
        return self.config.name

    @property
    def description(self) -> str:
        """Описание агента"""
        return self.config.description

    @abstractmethod
    async def process(
            self,
            input_data: Dict[str, Any],
            assessment_id: str
    ) -> AgentTaskResult:
        """Основной метод обработки для агента"""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Получение системного промпта для агента"""
        pass
    
    async def run(
        self, 
        input_data: Dict[str, Any], 
        assessment_id: str
    ) -> AgentTaskResult:
        """
        Запуск агента с обработкой ошибок и повторами
        
        Args:
            input_data: Входные данные
            assessment_id: ID оценки
            
        Returns:
            Результат выполнения
        """
        task_result = AgentTaskResult(
            agent_name=self.name,
            task_type=self._get_task_type(),
            status=ProcessingStatus.IN_PROGRESS,
            start_time=datetime.now()
        )
        
        # Логируем начало работы
        self.logger.log_agent_start(self.name, self._get_task_type(), assessment_id)
        
        for attempt in range(self.config.max_retries):
            try:
                # Выполняем основную обработку
                result = await asyncio.wait_for(
                    self.process(input_data, assessment_id),
                    timeout=self.config.timeout_seconds
                )
                
                # Обновляем статистику
                self._update_stats(True, result.execution_time_seconds or 0)
                
                # Логируем успех
                self.logger.log_agent_success(
                    self.name, 
                    self._get_task_type(), 
                    assessment_id, 
                    result.execution_time_seconds or 0
                )
                
                return result
                
            except asyncio.TimeoutError:
                error_msg = f"Тайм-аут выполнения ({self.config.timeout_seconds}с)"
                await self._handle_retry(task_result, error_msg, attempt, assessment_id)
                
            except Exception as e:
                error_msg = f"Ошибка выполнения: {str(e)}"
                await self._handle_retry(task_result, error_msg, attempt, assessment_id)
        
        # Все попытки исчерпаны
        task_result.status = ProcessingStatus.FAILED
        task_result.end_time = datetime.now()
        task_result.execution_time_seconds = (
            task_result.end_time - task_result.start_time
        ).total_seconds()
        
        self._update_stats(False, task_result.execution_time_seconds)
        
        self.logger.log_agent_error(
            self.name, 
            self._get_task_type(), 
            assessment_id, 
            Exception(task_result.error_message or "Неизвестная ошибка")
        )
        
        return task_result
    
    async def call_llm(
        self,
        system_prompt: str,
        user_message: str,
        context: Optional[str] = None,
        assessment_id: str = "unknown",
        temperature: Optional[float] = None
    ) -> str:
        """
        Вызов LLM с логированием и обработкой ошибок
        
        Args:
            system_prompt: Системный промпт
            user_message: Сообщение пользователя
            context: Дополнительный контекст
            assessment_id: ID оценки для логирования
            temperature: Температура генерации
            
        Returns:
            Ответ от LLM
        """
        messages = [
            LLMMessage(role="system", content=system_prompt)
        ]
        
        # Добавляем контекст если есть
        if context:
            messages.append(
                LLMMessage(
                    role="user", 
                    content=f"Контекст:\n{context}\n\nЗадача:\n{user_message}"
                )
            )
        else:
            messages.append(LLMMessage(role="user", content=user_message))
        
        # Вызываем LLM
        response = await self.llm_client.complete_chat(
            messages=messages,
            temperature=temperature or self.config.temperature
        )
        
        # Логируем вызов LLM
        self.logger.log_llm_request(
            self.name,
            assessment_id,
            response.model,
            response.usage.get("total_tokens", 0)
        )
        
        return response.content

    def _create_structured_fallback(self, expected_format: str, error_message: str) -> Dict[str, Any]:
        """Создание fallback результата для структурированных данных"""

        if expected_format.upper() == "JSON":
            # Проверяем контекст по ошибке или классу агента
            if "профил" in error_message.lower() or "profile" in error_message.lower() or self.name == "enhanced_profiler":
                return {
                    "name": "Unknown Agent",
                    "version": "1.0",
                    "description": f"ИИ-агент с автоматически сгенерированным описанием. Причина: {error_message[:100]}",
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
                        "overview": f"Базовый обзор агента. Ошибка анализа: {error_message[:100]}",
                        "technical_architecture": f"Техническая архитектура не определена из-за ошибки LLM: {error_message[:100]}",
                        "operational_model": f"Операционная модель не определена из-за ошибки LLM: {error_message[:100]}"
                    }
                }
            elif "риск" in error_message.lower() or "risk" in error_message.lower():
                return {
                    "probability_score": 3,
                    "impact_score": 3,
                    "total_score": 9,
                    "risk_level": "medium",
                    "probability_reasoning": f"Fallback оценка из-за ошибки LLM: {error_message[:200]}",
                    "impact_reasoning": f"Fallback оценка из-за ошибки LLM: {error_message[:200]}",
                    "key_factors": ["Ошибка получения данных от LLM"],
                    "recommendations": ["Проверить качество промпта", "Повторить оценку", "Проверить настройки LLM"],
                    "confidence_level": 0.1
                }
            else:
                # Общий fallback
                return {
                    "error": "Ошибка парсинга LLM ответа",
                    "error_message": error_message,
                    "fallback_response": True,
                    "timestamp": datetime.now().isoformat()
                }

        return {"error": f"Неподдерживаемый формат: {expected_format}"}

    def _validate_structured_result(self, result: Any, expected_format: str, logger) -> bool:
        """Валидация структурированного результата от LLM"""

        if not result:
            logger.warning("❌ Пустой результат от LLM")
            return False

        if expected_format.upper() == "JSON":
            if not isinstance(result, dict):
                logger.warning(f"❌ Ожидался dict, получен {type(result)}")
                return False

            # Проверяем основные поля для профилирования
            if "name" in result:  # Это профиль агента
                required_fields = ["name", "description", "agent_type"]
                missing_fields = [field for field in required_fields if field not in result]
                if missing_fields:
                    logger.warning(f"❌ Отсутствуют обязательные поля: {missing_fields}")
                    return False

                # Проверяем длину описания
                description = result.get("description", "")
                if len(str(description)) < 100:
                    logger.warning(f"❌ Слишком короткое описание: {len(str(description))} символов")
                    return False

            # Проверяем для оценки рисков
            elif "probability_score" in result:  # Это оценка риска
                required_fields = ["probability_score", "impact_score", "risk_level"]
                missing_fields = [field for field in required_fields if field not in result]
                if missing_fields:
                    logger.warning(f"❌ Отсутствуют обязательные поля оценки: {missing_fields}")
                    return False

        logger.info("✅ Структурированный результат прошел валидацию")
        return True

    # 3. ИСПРАВЛЕНИЕ: Логирование статистики результата
    def _log_result_statistics(self, result: Dict[str, Any], logger):
        """Логирование статистики результата"""

        if not isinstance(result, dict):
            return

        # Общие статистики
        logger.info(f"📊 Получен результат с {len(result)} полями")

        # Статистики для профиля агента
        if "description" in result:
            desc_len = len(str(result["description"]))
            logger.info(f"📝 Длина описания: {desc_len} символов")

            if desc_len < 200:
                logger.warning(f"⚠️ Короткое описание: {desc_len} < 200 символов")
            elif desc_len >= 500:
                logger.info(f"🎯 Отличная длина описания: {desc_len} >= 500 символов")

        # Статистики для detailed_summary
        if "detailed_summary" in result and isinstance(result["detailed_summary"], dict):
            summary = result["detailed_summary"]
            for section, content in summary.items():
                content_len = len(str(content))
                logger.info(f"📋 {section}: {content_len} символов")

                if content_len < 100:
                    logger.warning(f"⚠️ Короткая секция {section}: {content_len} < 100")

        # Статистики для списков
        for field in ["system_prompts", "guardrails", "external_apis"]:
            if field in result and isinstance(result[field], list):
                count = len(result[field])
                logger.info(f"📋 {field}: {count} элементов")

    async def call_llm_structured(
            self,
            data_to_analyze: str,
            extraction_prompt: str,
            assessment_id: str,
            expected_format: str = "JSON"
    ) -> Dict[str, Any]:
        """Вызов LLM для получения структурированных данных с улучшенной обработкой ошибок"""

        if not isinstance(self.llm_client, (LLMClient, RiskAnalysisLLMClient)):
            raise ValueError("LLM клиент не поддерживает структурированные запросы")

        # 🔍 ДИАГНОСТИКА: Проверяем длину промпта
        prompt_length = len(extraction_prompt)
        data_length = len(data_to_analyze)
        bound_logger = self.logger.bind_context(assessment_id, self.name)

        bound_logger.info(f"📏 Длина системного промпта: {prompt_length} символов")
        bound_logger.info(f"📏 Длина данных для анализа: {data_length} символов")
        bound_logger.info(f"📏 Общий размер контекста: {prompt_length + data_length} символов")

        # Проверяем качество промпта
        if prompt_length < 3000:
            bound_logger.warning(f"⚠️ Системный промпт может быть слишком коротким для детальных требований")
        else:
            bound_logger.info(f"✅ Системный промпт достаточно длинный для детальных требований")

        # Проверяем требования к длине в промпте
        if "МИНИМУМ" in extraction_prompt and ("символов" in extraction_prompt or "слов" in extraction_prompt):
            bound_logger.info("✅ В промпте найдены требования к минимальной длине рассуждений")
        else:
            bound_logger.warning("⚠️ В промпте НЕ найдены требования к длине рассуждений")

        # ИСПРАВЛЕНО: Множественные попытки с разными стратегиями
        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                bound_logger.info(f"🤖 LLM запрос (попытка {attempt + 1}/{max_attempts})...")

                # Вариация температуры и промпта для повторных попыток
                temperature = 0.05 if attempt == 0 else 0.1 if attempt == 1 else 0.15

                # Для повторных попыток добавляем усиления в промпт
                enhanced_prompt = extraction_prompt
                if attempt > 0:
                    enhanced_prompt += f"\n\n❗ ВНИМАНИЕ: Попытка {attempt + 1}. Предыдущие попытки не удались. ОБЯЗАТЕЛЬНО верни валидный JSON без ошибок!"

                result = await self.llm_client.extract_structured_data(
                    data_to_analyze=data_to_analyze,
                    extraction_prompt=enhanced_prompt,
                    expected_format=expected_format
                )

                # ИСПРАВЛЕНО: Дополнительная валидация результата
                if self._validate_structured_result(result, expected_format, bound_logger):
                    bound_logger.info(f"✅ LLM вернул валидный результат на попытке {attempt + 1}")

                    # Логируем статистику контента
                    self._log_result_statistics(result, bound_logger)

                    return result
                else:
                    raise ValueError(f"LLM вернул невалидную структуру данных на попытке {attempt + 1}")

            except Exception as e:
                last_error = e
                bound_logger.warning(f"⚠️ Попытка {attempt + 1} не удалась: {str(e)[:100]}...")

                if attempt < max_attempts - 1:
                    import asyncio
                    await asyncio.sleep(1 + attempt)  # Прогрессивная задержка

        # Все попытки не удались - создаем fallback результат
        bound_logger.error(f"❌ Все {max_attempts} попыток LLM запроса не удались. Последняя ошибка: {last_error}")

        fallback_result = self._create_structured_fallback(expected_format, str(last_error))
        bound_logger.warning(f"🔧 Создан fallback результат: {type(fallback_result)}")

        return fallback_result
    
    def validate_result(self, result_data: Dict[str, Any]) -> bool:
        """
        Базовая валидация результата агента
        Переопределяется в наследниках для специфичной валидации
        
        Args:
            result_data: Данные результата для валидации
            
        Returns:
            True если результат валиден
        """
        # Базовая проверка
        if not isinstance(result_data, dict):
            return False
        
        # Проверяем обязательные поля (переопределяется в наследниках)
        required_fields = self._get_required_result_fields()
        for field in required_fields:
            if field not in result_data:
                return False
        
        return True
    
    def _get_task_type(self) -> str:
        """Получение типа задачи агента"""
        return self.__class__.__name__.lower().replace('agent', '')
    
    def _get_required_result_fields(self) -> List[str]:
        """Получение списка обязательных полей результата (переопределяется в наследниках)"""
        return []
    
    async def _handle_retry(
        self, 
        task_result: AgentTaskResult, 
        error_msg: str, 
        attempt: int, 
        assessment_id: str
    ):
        """Обработка повторной попытки"""
        task_result.error_message = error_msg
        
        if attempt < self.config.max_retries - 1:
            # Логируем повтор
            self.logger.log_agent_retry(
                self.name, 
                self._get_task_type(), 
                assessment_id, 
                attempt + 1
            )
            
            # Небольшая задержка перед повтором
            await asyncio.sleep(1.0 * (attempt + 1))
        else:
            # Последняя попытка - фиксируем ошибку
            task_result.status = ProcessingStatus.FAILED
            task_result.end_time = datetime.now()
    
    def _update_stats(self, success: bool, execution_time: float):
        """Обновление статистики агента"""
        self.stats["total_requests"] += 1
        self.stats["total_execution_time"] += execution_time
        
        if success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # Пересчитываем среднее время ответа
        self.stats["average_response_time"] = (
            self.stats["total_execution_time"] / self.stats["total_requests"]
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики работы агента"""
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_requests"] / max(self.stats["total_requests"], 1)
            ),
            "agent_name": self.name,
            "agent_type": self._get_task_type()
        }
    
    async def health_check(self) -> bool:
        """Проверка работоспособности агента"""
        try:
            # Проверяем доступность LLM
            llm_healthy = await self.llm_client.health_check()
            
            # Можно добавить дополнительные проверки
            return llm_healthy
            
        except Exception:
            return False
    
    async def cleanup(self):
        """Очистка ресурсов агента"""
        try:
            await self.llm_client.close()
        except Exception:
            pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
    
    def __repr__(self) -> str:
        return self.__str__()


class AnalysisAgent(BaseAgent):
    """
    Базовый класс для агентов анализа
    Расширяет BaseAgent функциональностью для анализа данных
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
    
    async def analyze_data(
        self,
        data: str,
        analysis_type: str,
        criteria: str,
        assessment_id: str,
        examples: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Общий метод анализа данных
        
        Args:
            data: Данные для анализа
            analysis_type: Тип анализа
            criteria: Критерии анализа
            assessment_id: ID оценки
            examples: Примеры для контекста
            
        Returns:
            Результат анализа
        """
        system_prompt = self.get_system_prompt()
        
        if examples:
            system_prompt += f"\n\nПРИМЕРЫ:\n{examples}"
        
        user_message = f"""ДАННЫЕ ДЛЯ АНАЛИЗА:
{data}

ТИП АНАЛИЗА: {analysis_type}

КРИТЕРИИ:
{criteria}

Выполни анализ согласно указанным критериям."""
        
        response = await self.call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            assessment_id=assessment_id
        )
        
        return {"analysis_result": response, "analysis_type": analysis_type}


class EvaluationAgent(BaseAgent):
    """Базовый класс для агентов-оценщиков рисков"""

    def __init__(self, config: AgentConfig):
        # КРИТИЧНО: Устанавливаем флаг ПЕРЕД вызовом super().__init__
        config.use_risk_analysis_client = True
        super().__init__(config)

        # ИСПРАВЛЕНО: Проверяем что создался правильный тип клиента
        expected_types = (DeepSeekRiskAnalysisLLMClient, GigaChatRiskAnalysisLLMClient, RiskAnalysisLLMClient)

        if not isinstance(self.llm_client, expected_types):
            raise ValueError(
                f"EvaluationAgent требует RiskAnalysisLLMClient или его наследников, "
                f"получен: {type(self.llm_client)}"
            )

        print(f"✅ EvaluationAgent создан с клиентом: {type(self.llm_client).__name__}")

    async def evaluate_risk(
            self,
            risk_type: str,
            agent_data: str,
            evaluation_criteria: str,
            assessment_id: str,
            examples: Optional[str] = None
    ) -> Dict[str, Any]:
        """Оценка риска с использованием специализированного клиента"""

        try:
            # ИСПРАВЛЕНО: Проверяем интерфейс клиента
            if not hasattr(self.llm_client, 'evaluate_risk'):
                raise ValueError(f"LLM клиент {type(self.llm_client)} не поддерживает evaluate_risk")

            # Вызываем метод оценки риска
            result = await self.llm_client.evaluate_risk(
                risk_type=risk_type,
                agent_data=agent_data,
                evaluation_criteria=evaluation_criteria,
                examples=examples
            )

            # Валидация результата
            validated_result = self._ensure_required_fields(result)

            # Логируем оценку
            self.logger.log_risk_evaluation(
                self.name,
                assessment_id,
                risk_type,
                validated_result["total_score"],
                validated_result["risk_level"]
            )

            return validated_result

        except Exception as e:
            # В случае ошибки возвращаем безопасные дефолтные данные
            self.logger.bind_context(assessment_id, self.name).error(
                f"❌ Ошибка оценки риска {risk_type}: {e}"
            )

            return self._get_default_evaluation_data(f"Ошибка оценки риска: {str(e)}")
    
    def _get_required_result_fields(self) -> List[str]:
        """Обязательные поля для результата оценки риска"""
        return [
            "probability_score", "impact_score", "total_score", 
            "risk_level", "probability_reasoning", "impact_reasoning"
        ]
    def _parse_llm_response(self, response_content: str) -> Dict[str, Any]:
        """ПОЛНОСТЬЮ ИСПРАВЛЕННЫЙ парсинг ответа LLM с максимальной надежностью"""
        
        try:
            # Шаг 1: Очистка контента
            cleaned_content = response_content.strip()
            
            # Удаляем теги <think>...</think> если есть
            import re
            cleaned_content = re.sub(r'<think>.*?</think>', '', cleaned_content, flags=re.DOTALL)
            cleaned_content = cleaned_content.strip()
            
            # Удаляем markdown блоки
            if "```json" in cleaned_content:
                # Находим все JSON блоки
                json_blocks = re.findall(r'```json\s*(.*?)\s*```', cleaned_content, re.DOTALL)
                if json_blocks:
                    json_content = json_blocks[-1].strip()  # Берем последний блок
                else:
                    # Если не найден закрывающий тег, берем все после ```json
                    start = cleaned_content.find("```json") + 7
                    json_content = cleaned_content[start:].strip()
            else:
                # Ищем JSON по фигурным скобкам
                json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
                if json_match:
                    json_content = json_match.group().strip()
                else:
                    json_content = cleaned_content
            
            # Шаг 2: Дополнительная очистка JSON
            # Убираем потенциальные концевые символы после }
            if '}' in json_content:
                end_pos = json_content.rfind('}')
                json_content = json_content[:end_pos + 1]
            
            # Шаг 3: Пытаемся парсить JSON
            try:
                parsed_data = json.loads(json_content)
            except json.JSONDecodeError:
                # Пытаемся починить распространенные ошибки JSON
                json_content = self._fix_common_json_errors(json_content)
                parsed_data = json.loads(json_content)
            
            # Шаг 4: КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ - всегда обеспечиваем обязательные поля
            parsed_data = self._ensure_required_fields(parsed_data)
            
            return parsed_data
            
        except Exception as e:
            # Если ничего не помогло, возвращаем безопасные дефолтные данные
            self.logger.bind_context("unknown", self.name).warning(
                f"⚠️ Критическая ошибка парсинга LLM ответа, используем fallback: {e}"
            )
            self.logger.bind_context("unknown", self.name).debug(
                f"Проблемный контент: {response_content[:200]}..."
            )
            return self._get_default_evaluation_data(f"Критическая ошибка парсинга: {str(e)}")

    def _fix_common_json_errors(self, json_content: str) -> str:
        """Исправляет распространенные ошибки в JSON от LLM"""
        
        # Убираем trailing commas
        json_content = re.sub(r',\s*}', '}', json_content)
        json_content = re.sub(r',\s*]', ']', json_content)
        
        # Исправляем неэкранированные кавычки в строках
        # Это простое исправление, может потребовать доработки
        json_content = re.sub(r'(?<!\\)"(?=[^,}\]]*[,}\]])', '\\"', json_content)
        
        # Убираем комментарии в JSON (если есть)
        json_content = re.sub(r'//.*?\n', '\n', json_content)
        json_content = re.sub(r'/\*.*?\*/', '', json_content, flags=re.DOTALL)
        
        return json_content

    def _ensure_required_fields(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обеспечение обязательных полей с дефолтными значениями"""

        required_fields = {
            "probability_score": 3,
            "impact_score": 3,
            "total_score": 9,
            "risk_level": "medium",
            "probability_reasoning": "Обоснование вероятности не предоставлено LLM",
            "impact_reasoning": "Обоснование воздействия не предоставлено LLM",
            "key_factors": [],
            "recommendations": [],
            "confidence_level": 0.7
        }

        # Добавляем отсутствующие поля
        for field, default_value in required_fields.items():
            if field not in parsed_data or parsed_data[field] is None:
                parsed_data[field] = default_value

        # Валидируем и исправляем типы данных
        parsed_data = self._validate_and_fix_field_types(parsed_data)

        # Валидируем бизнес-логику
        parsed_data = self._validate_business_logic(parsed_data)

        return parsed_data

    def _validate_numeric_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует и исправляет числовые поля"""
        
        # Валидация probability_score (1-5)
        try:
            data["probability_score"] = int(data["probability_score"])
            if not (1 <= data["probability_score"] <= 5):
                data["probability_score"] = 3
        except (ValueError, TypeError):
            data["probability_score"] = 3
        
        # Валидация impact_score (1-5)
        try:
            data["impact_score"] = int(data["impact_score"])
            if not (1 <= data["impact_score"] <= 5):
                data["impact_score"] = 3
        except (ValueError, TypeError):
            data["impact_score"] = 3
        
        # Валидация confidence_level (0.0-1.0)
        try:
            data["confidence_level"] = float(data["confidence_level"])
            if not (0.0 <= data["confidence_level"] <= 1.0):
                data["confidence_level"] = 0.7
        except (ValueError, TypeError):
            data["confidence_level"] = 0.7
        
        return data

    def _validate_string_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует и исправляет строковые поля"""
        
        # Валидация risk_level
        valid_levels = ["low", "medium", "high"]
        if data.get("risk_level") not in valid_levels:
            data["risk_level"] = "medium"
        
        # Валидация reasoning полей
        if not data.get("probability_reasoning") or len(str(data["probability_reasoning"]).strip()) < 10:
            data["probability_reasoning"] = "Обоснование вероятности не предоставлено или некорректно"
        
        if not data.get("impact_reasoning") or len(str(data["impact_reasoning"]).strip()) < 10:
            data["impact_reasoning"] = "Обоснование воздействия не предоставлено или некорректно"
        
        return data

    def _validate_list_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует и исправляет списковые поля"""
        
        list_fields = ["key_factors", "recommendations"]
        
        for field in list_fields:
            if not isinstance(data.get(field), list):
                data[field] = []
            
            # Убираем пустые строки и None
            data[field] = [
                item for item in data[field] 
                if item and isinstance(item, str) and item.strip()
            ]
            
            # Ограничиваем количество элементов
            data[field] = data[field][:10]
        
        return data

    def _validate_and_fix_field_types(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует и исправляет типы полей"""

        # Числовые поля (1-5)
        score_fields = ["probability_score", "impact_score"]
        for field in score_fields:
            try:
                value = int(data[field])
                data[field] = max(1, min(5, value))  # Ограничиваем диапазон 1-5
            except (ValueError, TypeError):
                data[field] = 3  # Средний балл

        # Confidence level (0.0-1.0)
        try:
            value = float(data["confidence_level"])
            data["confidence_level"] = max(0.0, min(1.0, value))
        except (ValueError, TypeError):
            data["confidence_level"] = 0.7

        # Risk level (enum)
        valid_levels = ["low", "medium", "high"]
        if data.get("risk_level") not in valid_levels:
            data["risk_level"] = "medium"

        # Строковые поля
        string_fields = ["probability_reasoning", "impact_reasoning"]
        for field in string_fields:
            if not isinstance(data.get(field), str) or len(str(data[field]).strip()) < 5:
                data[field] = f"Автоматически сгенерированное обоснование для {field}"

        # Списковые поля
        list_fields = ["key_factors", "recommendations"]
        for field in list_fields:
            if not isinstance(data.get(field), list):
                data[field] = []
            else:
                # Очищаем список от пустых элементов
                data[field] = [
                                  str(item).strip() for item in data[field]
                                  if item and str(item).strip()
                              ][:10]  # Ограничиваем до 10 элементов

        return data

    def _validate_business_logic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует бизнес-логику оценки риска"""

        # Пересчитываем total_score для консистентности
        data["total_score"] = data["probability_score"] * data["impact_score"]

        # Корректируем risk_level на основе total_score
        total_score = data["total_score"]
        if total_score <= 6:
            correct_level = "low"
        elif total_score <= 14:
            correct_level = "medium"
        else:
            correct_level = "high"

        if data["risk_level"] != correct_level:
            data["risk_level"] = correct_level

        return data

    def _get_default_evaluation_data(self, error_message: str) -> Dict[str, Any]:
        """Безопасные дефолтные данные для оценки"""
        return {
            "probability_score": 3,
            "impact_score": 3,
            "total_score": 9,
            "risk_level": "medium",
            "probability_reasoning": f"LLM не смог предоставить обоснование. Ошибка: {error_message}",
            "impact_reasoning": f"LLM не смог предоставить обоснование. Ошибка: {error_message}",
            "key_factors": ["Недостаточно данных для анализа"],
            "recommendations": ["Провести дополнительный анализ", "Улучшить качество входных данных"],
            "confidence_level": 0.3
        }

# ===============================
# Фабрики для создания агентов
# ===============================

def create_agent_config(
        name: str,
        description: str,
        llm_base_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        use_risk_analysis_client: bool = False
) -> AgentConfig:
    """Создание конфигурации агента с центральным конфигуратором"""

    # Получаем настройки из центрального конфигуратора
    manager = get_llm_config_manager()
    base_config = manager.get_config()

    # Используем значения из конфигуратора или переопределяем
    actual_base_url = llm_base_url or base_config.base_url
    actual_model = llm_model or base_config.model
    actual_temperature = temperature if temperature is not None else base_config.temperature
    actual_max_retries = max_retries if max_retries is not None else base_config.max_retries
    actual_timeout = timeout_seconds if timeout_seconds is not None else base_config.timeout

    # Создаем LLM конфигурацию со ВСЕМИ полями
    llm_config = LLMConfig(
        base_url=actual_base_url,
        model=actual_model,
        temperature=actual_temperature,
        max_tokens=base_config.max_tokens,
        timeout=actual_timeout,
        max_retries=actual_max_retries,
        retry_delay=base_config.retry_delay,
        provider=base_config.provider,
        cert_file=base_config.cert_file,
        key_file=base_config.key_file,
        top_p=base_config.top_p,
        verify_ssl_certs=base_config.verify_ssl_certs,
        profanity_check=base_config.profanity_check,
        streaming=base_config.streaming,
        api_key=base_config.api_key  # ДОБАВЛЕНО: API ключ для DeepSeek
    )

    return AgentConfig(
        name=name,
        description=description,
        llm_config=llm_config,
        max_retries=actual_max_retries,
        timeout_seconds=actual_timeout,
        temperature=actual_temperature,
        use_risk_analysis_client=use_risk_analysis_client
    )


def create_default_config_from_env() -> AgentConfig:
    """Создание конфигурации по умолчанию из центрального конфигуратора"""
    return create_agent_config(
        name="default_agent",
        description="Агент с настройками по умолчанию"
    )


# Экспорт основных классов
__all__ = [
    "BaseAgent",
    "AnalysisAgent", 
    "EvaluationAgent",
    "AgentConfig",
    "create_agent_config",
    "create_default_config_from_env"
]
