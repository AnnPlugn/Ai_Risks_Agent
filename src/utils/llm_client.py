"""
Клиент для работы с LLM моделями
Поддерживает qwen3-4b через LM Studio (localhost:1234), GigaChat и DeepSeek API
"""

import json
import asyncio

try:
    from langchain_gigachat import GigaChat

    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False
    GigaChat = None

try:
    from openai import AsyncOpenAI

    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    AsyncOpenAI = None

from .llm_config_manager import LLMProvider, LLMConfig, get_llm_config_manager
from typing import Dict, List, Optional, Any, AsyncGenerator

from dataclasses import dataclass
from datetime import datetime

import httpx
from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """Сообщение для LLM"""
    role: str = Field(..., description="Роль: system, user, assistant")
    content: str = Field(..., description="Содержание сообщения")


class LLMResponse(BaseModel):
    """Ответ от LLM"""
    content: str = Field(..., description="Содержание ответа")
    finish_reason: str = Field(..., description="Причина завершения")
    usage: Dict[str, int] = Field(default_factory=dict, description="Статистика использования")
    model: str = Field(..., description="Использованная модель")
    created: datetime = Field(default_factory=datetime.now, description="Время создания")


class LLMError(Exception):
    """Исключение при работе с LLM"""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class LLMClient:
    """Асинхронный клиент для работы с LLM через OpenAI-совместимый API"""

    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            try:
                self.config = LLMConfig.from_manager()
            except Exception as e:
                raise LLMError(
                    f"Ошибка загрузки конфигурации LLM: {str(e)}. "
                    f"Проверьте переменные окружения LLM_PROVIDER, GIGACHAT_CERT_PATH, GIGACHAT_KEY_PATH, LLM_DeepSeek_API_KEY"
                ) from e
        else:
            self.config = config

        if self.config.provider == LLMProvider.GIGACHAT:
            self.client = None
        else:
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout
            )

        # Статистика
        self.total_requests = 0
        self.total_tokens = 0
        self.error_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Закрытие клиента"""
        if self.client:
            await self.client.aclose()

    async def health_check(self) -> bool:
        """Проверка доступности LLM сервера"""
        try:
            response = await self.client.get("/v1/models", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    async def get_available_models(self) -> List[str]:
        """Получение списка доступных моделей"""
        try:
            response = await self.client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except Exception as e:
            raise LLMError(f"Ошибка получения списка моделей: {str(e)}")

    async def complete_chat(
            self,
            messages: List[LLMMessage],
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            stream: bool = False
    ) -> LLMResponse:
        """
        Выполнение chat completion запроса
        """
        request_data = {
            "model": model or self.config.model,
            "messages": [msg.dict() for msg in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": stream
        }

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.post(
                    "/v1/chat/completions",
                    json=request_data,
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                self.total_requests += 1
                self.total_tokens += usage.get("total_tokens", 0)
                return LLMResponse(
                    content=choice["message"]["content"],
                    finish_reason=choice["finish_reason"],
                    usage=usage,
                    model=data["model"],
                    created=datetime.now()
                )
            except httpx.HTTPStatusError as e:
                self.error_count += 1
                error_msg = f"HTTP {e.response.status_code}: {e.response.text}"
                if attempt == self.config.max_retries - 1:
                    raise LLMError(
                        f"Ошибка запроса к LLM после {self.config.max_retries} попыток: {error_msg}",
                        status_code=e.response.status_code,
                        response_data=e.response.json() if e.response.text else None
                    )
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
            except Exception as e:
                self.error_count += 1
                if attempt == self.config.max_retries - 1:
                    raise LLMError(f"Неожиданная ошибка: {str(e)}")
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

    async def complete_chat_stream(
            self,
            messages: List[LLMMessage],
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Потоковый chat completion
        """
        request_data = {
            "model": model or self.config.model,
            "messages": [msg.dict() for msg in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": True
        }

        try:
            async with self.client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json=request_data
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choice = data["choices"][0]
                            if "delta" in choice and "content" in choice["delta"]:
                                content = choice["delta"]["content"]
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise LLMError(f"Ошибка потокового запроса: {str(e)}")

    async def analyze_with_prompt(
            self,
            system_prompt: str,
            user_input: str,
            context: Optional[str] = None,
            model: Optional[str] = None,
            temperature: Optional[float] = None
    ) -> LLMResponse:
        """
        Упрощенный метод для анализа с промптом
        """
        messages = [LLMMessage(role="system", content=system_prompt)]
        if context:
            messages.append(
                LLMMessage(
                    role="user",
                    content=f"Контекст:\n{context}\n\nЗадача:\n{user_input}"
                )
            )
        else:
            messages.append(LLMMessage(role="user", content=user_input))
        return await self.complete_chat(
            messages=messages,
            model=model,
            temperature=temperature
        )

    async def extract_structured_data(
            self,
            data_to_analyze: str,
            extraction_prompt: str,
            expected_format: str = "JSON",
            model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Извлечение структурированных данных"""
        system_prompt = f"""Ты - эксперт по анализу данных. 

        Твоя задача: {extraction_prompt}

        КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ:
        - Отвечай ТОЛЬКО валидным {expected_format} без дополнительного текста
        - НЕ добавляй комментарии, пояснения, теги <think> или markdown блоки
        - Если данных недостаточно, используй разумные значения по умолчанию
        - Строго следуй указанной структуре данных
        - Все числовые поля ОБЯЗАТЕЛЬНО должны быть числами, не строками
        - Все обязательные поля должны присутствовать
        - НЕ используй запятые в конце объектов или массивов

        ПРИМЕР ПРАВИЛЬНОГО ФОРМАТА:
        {{
            "probability_score": 3,
            "impact_score": 4,
            "total_score": 12,
            "risk_level": "medium",
            "probability_reasoning": "Обоснование вероятности",
            "impact_reasoning": "Обоснование воздействия",
            "key_factors": ["фактор1", "фактор2"],
            "recommendations": ["рекомендация1", "рекомендация2"],
            "confidence_level": 0.8
        }}

        СТРОГО: отвечай только JSON, начинающийся с {{ и заканчивающийся }}"""
        max_retries = 4
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await self.analyze_with_prompt(
                    system_prompt=system_prompt,
                    user_input=f"Данные для анализа:\n{data_to_analyze}",
                    model=model,
                    temperature=0.05 if attempt == 0 else 0.1
                )
                parsed_result = self._ultra_robust_json_parser(response.content)
                validated_result = self._validate_and_fix_json_structure(parsed_result)
                return validated_result
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    system_prompt += f"\n\nВНИМАНИЕ: Попытка {attempt + 1} из {max_retries}. Предыдущая ошибка: {str(e)[:100]}. Будь ОСОБЕННО внимательным к формату JSON!"
                    await asyncio.sleep(1 + attempt)
                else:
                    return self._create_emergency_fallback_result(extraction_prompt, str(e))
        return self._create_emergency_fallback_result(extraction_prompt, str(last_error))

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики использования"""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "error_count": self.error_count,
            "success_rate": (self.total_requests - self.error_count) / max(self.total_requests, 1),
            "avg_tokens_per_request": self.total_tokens / max(self.total_requests, 1)
        }

    def _ultra_robust_json_parser(self, content: str) -> Dict[str, Any]:
        """Максимально надежный парсер JSON"""
        import re
        import json
        cleaned = content.strip()
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL).strip()
        if '```json' in cleaned:
            json_blocks = re.findall(r'```json\s*(.*?)\s*```', cleaned, re.DOTALL)
            if json_blocks:
                cleaned = json_blocks[-1].strip()
            else:
                start = cleaned.find('```json') + 7
                cleaned = cleaned[start:].strip()
        cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL).strip()
        strategies = [
            lambda x: json.loads(x),
            lambda x: json.loads(self._extract_json_by_braces(x)),
            lambda x: json.loads(self._extract_json_by_regex(x)),
            lambda x: json.loads(self._fix_common_json_issues(x)),
            lambda x: json.loads(self._aggressive_json_fix(x))
        ]
        for i, strategy in enumerate(strategies):
            try:
                result = strategy(cleaned)
                if isinstance(result, dict):
                    return result
            except Exception as e:
                if i == len(strategies) - 1:
                    raise Exception(f"Все стратегии парсинга не удались. Последняя ошибка: {e}")
                continue
        raise Exception("Невозможно распарсить JSON ни одной стратегией")

    def _extract_json_by_braces(self, content: str) -> str:
        start = content.find('{')
        if start == -1:
            raise ValueError("Не найдена открывающая фигурная скобка")
        brace_count = 0
        end = start
        for i, char in enumerate(content[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break
        if brace_count != 0:
            end = len(content) - 1
        return content[start:end + 1]

    def _extract_json_by_regex(self, content: str) -> str:
        import re
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, content, re.DOTALL)
        if matches:
            return max(matches, key=len)
        raise ValueError("JSON объект не найден регулярным выражением")

    def _fix_common_json_issues(self, content: str) -> str:
        import re
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        content = re.sub(r"'([^']*)':", r'"\1":', content)
        content = re.sub(r":\s*'([^']*)'", r': "\1"', content)
        content = re.sub(r':\s*([^",{\[\]\s][^,}\]]*[^",}\]\s])\s*[,}]',
                         lambda m: f': "{m.group(1).strip()}"' + m.group(0)[-1], content)
        content = re.sub(r'(?<!\\)"(?=[^,}\]]*[,}\]])', '\\"', content)
        content = re.sub(r'//.*?\n', '\n', content)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content = re.sub(r'\s+', ' ', content)
        return content.strip()

    def _aggressive_json_fix(self, content: str) -> str:
        import re
        content = self._fix_common_json_issues(content)
        if not content.strip().startswith('{'):
            content = '{' + content
        if not content.strip().endswith('}'):
            content = content + '}'
        lines = content.split('\n')
        fixed_lines = []
        for line in lines:
            quote_count = line.count('"') - line.count('\\"')
            if quote_count % 2 == 1:
                if ':' in line and not line.strip().endswith('"'):
                    line = line.rstrip() + '"'
            fixed_lines.append(line)
        content = '\n'.join(fixed_lines)
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            content = content[start:end + 1]
        return content

    def _validate_and_fix_json_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self._looks_like_risk_evaluation(data):
            return self._fix_risk_evaluation_structure(data)
        elif self._looks_like_critic_evaluation(data):
            return self._fix_critic_evaluation_structure(data)
        else:
            return self._fix_general_structure(data)

    def _looks_like_risk_evaluation(self, data: Dict[str, Any]) -> bool:
        risk_fields = {"probability_score", "impact_score", "total_score", "risk_level"}
        return any(field in data for field in risk_fields)

    def _looks_like_critic_evaluation(self, data: Dict[str, Any]) -> bool:
        critic_fields = {"quality_score", "is_acceptable", "critic_reasoning"}
        return any(field in data for field in critic_fields)

    def _fix_risk_evaluation_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = {
            "probability_score": 3,
            "impact_score": 3,
            "total_score": 9,
            "risk_level": "medium",
            "probability_reasoning": "Обоснование не предоставлено",
            "impact_reasoning": "Обоснование не предоставлено",
            "key_factors": [],
            "recommendations": [],
            "confidence_level": 0.7
        }
        for field, default_value in required_fields.items():
            if field not in data or data[field] is None:
                data[field] = default_value
        data["probability_score"] = self._ensure_int_range(data["probability_score"], 1, 5, 3)
        data["impact_score"] = self._ensure_int_range(data["impact_score"], 1, 5, 3)
        data["confidence_level"] = self._ensure_float_range(data["confidence_level"], 0.0, 1.0, 0.7)
        data["total_score"] = data["probability_score"] * data["impact_score"]
        total_score = data["total_score"]
        if total_score <= 6:
            data["risk_level"] = "low"
        elif total_score <= 14:
            data["risk_level"] = "medium"
        else:
            data["risk_level"] = "high"
        data["key_factors"] = self._ensure_string_list(data["key_factors"])
        data["recommendations"] = self._ensure_string_list(data["recommendations"])
        data["probability_reasoning"] = self._ensure_string(data["probability_reasoning"],
                                                            "Обоснование не предоставлено")
        data["impact_reasoning"] = self._ensure_string(data["impact_reasoning"], "Обоснование не предоставлено")
        return data

    def _fix_critic_evaluation_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = {
            "quality_score": 5.0,
            "is_acceptable": True,
            "issues_found": [],
            "improvement_suggestions": [],
            "critic_reasoning": "Обоснование не предоставлено"
        }
        for field, default_value in required_fields.items():
            if field not in data or data[field] is None:
                data[field] = default_value
        data["quality_score"] = self._ensure_float_range(data["quality_score"], 0.0, 10.0, 5.0)
        data["is_acceptable"] = bool(data.get("is_acceptable", True))
        data["issues_found"] = self._ensure_string_list(data["issues_found"])
        data["improvement_suggestions"] = self._ensure_string_list(data["improvement_suggestions"])
        data["critic_reasoning"] = self._ensure_string(data["critic_reasoning"], "Обоснование не предоставлено")
        return data

    def _fix_general_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {"error": "Неверный формат данных", "original_data": str(data)}
        return data

    def _ensure_int_range(self, value: Any, min_val: int, max_val: int, default: int) -> int:
        try:
            int_val = int(float(value))
            return max(min_val, min(max_val, int_val))
        except (ValueError, TypeError):
            return default

    def _ensure_float_range(self, value: Any, min_val: float, max_val: float, default: float) -> float:
        try:
            float_val = float(value)
            return max(min_val, min(max_val, float_val))
        except (ValueError, TypeError):
            return default

    def _ensure_string(self, value: Any, default: str) -> str:
        if not value or not isinstance(value, str) or len(value.strip()) < 3:
            return default
        return str(value).strip()

    def _ensure_string_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            if item and isinstance(item, str) and len(item.strip()) > 0:
                result.append(str(item).strip())
        return result[:10]

    def _create_emergency_fallback_result(self, extraction_prompt: str, error_message: str) -> Dict[str, Any]:
        prompt_lower = extraction_prompt.lower()
        if any(keyword in prompt_lower for keyword in ['риск', 'risk', 'оцен', 'evaluat']):
            return {
                "probability_score": 3,
                "impact_score": 3,
                "total_score": 9,
                "risk_level": "medium",
                "probability_reasoning": f"Аварийная оценка: LLM вернул некорректный JSON. Ошибка: {error_message}",
                "impact_reasoning": f"Аварийная оценка: LLM вернул некорректный JSON. Ошибка: {error_message}",
                "key_factors": ["Ошибка парсинга ответа LLM"],
                "recommendations": ["Проверить качество промпта", "Повторить оценку", "Проверить настройки LLM"],
                "confidence_level": 0.1
            }
        elif any(keyword in prompt_lower for keyword in ['критик', 'critic', 'качеств', 'quality']):
            return {
                "quality_score": 3.0,
                "is_acceptable": False,
                "issues_found": ["LLM вернул некорректный JSON", f"Ошибка парсинга: {error_message}"],
                "improvement_suggestions": ["Улучшить промпт", "Проверить настройки LLM", "Повторить оценку"],
                "critic_reasoning": f"Аварийная оценка качества: не удалось распарсить ответ LLM. Ошибка: {error_message}"
            }
        else:
            return {
                "error": "Ошибка парсинга LLM ответа",
                "error_message": error_message,
                "extraction_prompt": extraction_prompt,
                "fallback_response": True
            }


class DeepSeekLLMClient(LLMClient):
    """Специализированный клиент для работы с DeepSeek через OpenAI API"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_manager()
        if not DEEPSEEK_AVAILABLE:
            raise ImportError("openai не установлен! Установите: pip install openai")
        if self.config.provider != LLMProvider.DEEPSEEK:
            raise ValueError("DeepSeekLLMClient требует provider=DEEPSEEK")
        if not self.config.api_key:
            raise ValueError("Для DeepSeek необходим API ключ (LLM_DeepSeek_API_KEY)")

        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        self.total_requests = 0
        self.total_tokens = 0
        self.error_count = 0

    async def health_check(self) -> bool:
        """Проверка доступности DeepSeek API"""
        try:
            response = await self.client.models.list()
            return bool(response.data)
        except Exception as e:
            print(f"❌ Ошибка проверки DeepSeek: {e}")
            return False

    async def get_available_models(self) -> List[str]:
        """Получение списка доступных моделей для DeepSeek"""
        try:
            response = await self.client.models.list()
            return [model.id for model in response.data]
        except Exception as e:
            raise LLMError(f"Ошибка получения списка моделей DeepSeek: {str(e)}")

    async def complete_chat(
            self,
            messages: List[LLMMessage],
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            stream: bool = False
    ) -> LLMResponse:
        """Выполнение chat completion через DeepSeek"""
        try:
            self.total_requests += 1
            response = await self.client.chat.completions.create(
                model=model or self.config.model,
                messages=[msg.dict() for msg in messages],
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                stream=stream
            )
            if stream:
                raise NotImplementedError("Потоковый режим пока не поддерживается для DeepSeek")
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            } if response.usage else {}
            self.total_tokens += usage.get("total_tokens", 0)
            return LLMResponse(
                content=content,
                finish_reason=finish_reason,
                usage=usage,
                model=response.model or (model or self.config.model),
                created=datetime.fromtimestamp(response.created)
            )
        except Exception as e:
            self.error_count += 1
            raise LLMError(f"Ошибка DeepSeek: {str(e)}")

    async def complete_chat_stream(
            self,
            messages: List[LLMMessage],
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """Потоковый chat completion для DeepSeek"""
        try:
            self.total_requests += 1
            async with self.client.chat.completions.create(
                    model=model or self.config.model,
                    messages=[msg.dict() for msg in messages],
                    temperature=temperature or self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                    stream=True
            ) as stream:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
        except Exception as e:
            self.error_count += 1
            raise LLMError(f"Ошибка потокового запроса DeepSeek: {str(e)}")

    async def close(self):
        """Закрытие клиента DeepSeek"""
        if self.client:
            await self.client.close()


class DeepSeekRiskAnalysisLLMClient(DeepSeekLLMClient):
    """Специализированный DeepSeek клиент для анализа рисков"""

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        if self.config.temperature > 0.3:
            self.config.temperature = 0.2

    async def evaluate_risk(
            self,
            risk_type: str,
            agent_data: str,
            evaluation_criteria: str,
            examples: Optional[str] = None
    ) -> Dict[str, Any]:
        """Оценка риска с использованием DeepSeek"""
        system_prompt = f"""Ты - эксперт по оценке операционных рисков ИИ-агентов в банковской сфере.

Твоя задача: оценить {risk_type} для предоставленного ИИ-агента.

КРИТЕРИИ ОЦЕНКИ:
{evaluation_criteria}

ШКАЛА ОЦЕНКИ:
- Вероятность: 1-5 баллов (1=низкая, 5=высокая)
- Тяжесть: 1-5 баллов (1=низкие потери, 5=высокие потери)
- Итоговый балл = Вероятность × Тяжесть

ФОРМАТ ОТВЕТА (СТРОГО JSON):
{{
    "probability_score": <1-5>,
    "impact_score": <1-5>,
    "total_score": <1-25>,
    "risk_level": "<low|medium|high>",
    "probability_reasoning": "<подробное обоснование вероятности>",
    "impact_reasoning": "<подробное обоснование тяжести>",
    "key_factors": ["<фактор1>", "<фактор2>", ...],
    "recommendations": ["<рекомендация1>", "<рекомендация2>", ...],
    "confidence_level": <0.0-1.0>
}}

УРОВНИ РИСКА:
- low: 1-6 баллов
- medium: 7-14 баллов  
- high: 15-25 баллов"""
        if examples:
            system_prompt += f"\n\nПРИМЕРЫ ОЦЕНОК:\n{examples}"
        response = await self.extract_structured_data(
            data_to_analyze=agent_data,
            extraction_prompt=f"Оцени {risk_type} согласно методике",
            expected_format="JSON"
        )
        required_fields = [
            "probability_score", "impact_score", "total_score",
            "risk_level", "probability_reasoning", "impact_reasoning"
        ]
        for field in required_fields:
            if field not in response:
                raise LLMError(f"Отсутствует обязательное поле в ответе: {field}")
        if not (1 <= response["probability_score"] <= 5):
            raise LLMError(f"Некорректный probability_score: {response['probability_score']}")
        if not (1 <= response["impact_score"] <= 5):
            raise LLMError(f"Некорректный impact_score: {response['impact_score']}")
        response["total_score"] = response["probability_score"] * response["impact_score"]
        score = response["total_score"]
        if score <= 6:
            response["risk_level"] = "low"
        elif score <= 14:
            response["risk_level"] = "medium"
        else:
            response["risk_level"] = "high"
        return response

    async def critique_evaluation(
            self,
            risk_type: str,
            original_evaluation: Dict[str, Any],
            agent_data: str,
            quality_threshold: float = 7.0
    ) -> Dict[str, Any]:
        """Критика оценки другого агента"""
        system_prompt = f"""Ты - критик-эксперт по оценке качества анализа рисков ИИ-агентов.

Твоя задача: оценить качество предоставленной оценки {risk_type}.

КРИТЕРИИ КАЧЕСТВА:
1. Обоснованность оценок (соответствие данным агента)
2. Полнота анализа (учтены ли все аспекты)
3. Логичность рассуждений
4. Практичность рекомендаций
5. Соответствие методике оценки

ШКАЛА КАЧЕСТВА: 0-10 баллов
ПОРОГ ПРИЕМЛЕМОСТИ: {quality_threshold} баллов

ФОРМАТ ОТВЕТА (СТРОГО JSON):
{{
    "quality_score": <0.0-10.0>,
    "is_acceptable": <true|false>,
    "issues_found": ["<проблема1>", "<проблема2>", ...],
    "improvement_suggestions": ["<предложение1>", "<предложение2>", ...],
    "critic_reasoning": "<подробное обоснование оценки качества>"
}}"""
        evaluation_text = json.dumps(original_evaluation, ensure_ascii=False, indent=2)
        context = f"""ДАННЫЕ ОБ АГЕНТЕ:
{agent_data}

ОЦЕНКА ДЛЯ КРИТИКИ:
{evaluation_text}"""
        response = await self.extract_structured_data(
            data_to_analyze=context,
            extraction_prompt="Критически оцени качество представленной оценки риска",
            expected_format="JSON"
        )
        required_fields = ["quality_score", "is_acceptable", "critic_reasoning"]
        for field in required_fields:
            if field not in response:
                raise LLMError(f"Отсутствует обязательное поле: {field}")
        response["is_acceptable"] = response["quality_score"] >= quality_threshold
        return response


class GigaChatLLMClient(LLMClient):
    """Специализированный клиент для работы с GigaChat через langchain_gigachat"""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_manager()
        if not GIGACHAT_AVAILABLE:
            raise ImportError("langchain_gigachat не установлен! Установите: pip install langchain-gigachat")
        if self.config.provider != LLMProvider.GIGACHAT:
            raise ValueError("GigaChatLLMClient требует provider=GIGACHAT")
        if not (self.config.cert_file and self.config.key_file):
            raise ValueError("Для GigaChat необходимы cert_file и key_file")
        self.gigachat = GigaChat(
            base_url=self.config.base_url,
            cert_file=self.config.cert_file,
            key_file=self.config.key_file,
            model=self.config.model,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            verify_ssl_certs=self.config.verify_ssl_certs,
            profanity_check=self.config.profanity_check,
            streaming=self.config.streaming
        )
        self.client = None
        self.total_requests = 0
        self.total_tokens = 0
        self.error_count = 0

    async def health_check(self) -> bool:
        try:
            loop = asyncio.get_event_loop()

            def sync_test():
                return self.gigachat.invoke("Привет")

            response = await loop.run_in_executor(None, sync_test)
            return bool(hasattr(response, 'content') and response.content)
        except Exception:
            return False

    async def get_available_models(self) -> List[str]:
        return ["GigaChat", "GigaChat-Pro", "GigaChat-Max"]

    async def complete_chat(
            self,
            messages: List[LLMMessage],
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            stream: bool = False
    ) -> LLMResponse:
        try:
            self.total_requests += 1
            prompt = self._format_messages_for_gigachat(messages)
            original_temp = self.gigachat.temperature
            original_model = self.gigachat.model
            if temperature is not None:
                self.gigachat.temperature = temperature
            if model is not None:
                self.gigachat.model = model
            try:
                loop = asyncio.get_event_loop()

                def sync_invoke():
                    return self.gigachat.invoke(prompt)

                response = await loop.run_in_executor(None, sync_invoke)
                content = response.content if hasattr(response, 'content') else str(response)
                estimated_tokens = len(prompt.split()) + len(content.split())
                self.total_tokens += estimated_tokens
                return LLMResponse(
                    content=content,
                    finish_reason="stop",
                    usage={"total_tokens": estimated_tokens, "estimated": True},
                    model=model or self.config.model,
                    created=datetime.now()
                )
            finally:
                self.gigachat.temperature = original_temp
                self.gigachat.model = original_model
        except Exception as e:
            self.error_count += 1
            raise LLMError(f"Ошибка GigaChat: {str(e)}")

    def _format_messages_for_gigachat(self, messages: List[LLMMessage]) -> str:
        formatted_parts = []
        for message in messages:
            role = message.role
            content = message.content
            if role == "system":
                formatted_parts.append(f"Системная инструкция: {content}")
            elif role == "user":
                formatted_parts.append(f"Пользователь: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Ассистент: {content}")
            else:
                formatted_parts.append(content)
        return "\n\n".join(formatted_parts)

    async def simple_completion(self, prompt: str, **kwargs) -> str:
        try:
            loop = asyncio.get_event_loop()

            def sync_invoke():
                return self.gigachat.invoke(prompt)

            response = await loop.run_in_executor(None, sync_invoke)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            raise LLMError(f"Ошибка GigaChat простого запроса: {str(e)}")

    async def extract_structured_data(
            self,
            data_to_analyze: str,
            extraction_prompt: str,
            expected_format: str = "JSON",
            max_attempts: int = 3
    ) -> Dict[str, Any]:
        system_prompt = f"""Ты - эксперт по анализу данных. 

        Твоя задача: {extraction_prompt}

        КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ:
        - Отвечай ТОЛЬКО валидным {expected_format} без дополнительного текста
        - НЕ добавляй комментарии, пояснения, теги <think> или markdown блоки
        - Если данных недостаточно, используй разумные значения по умолчанию
        - Строго следуй указанной структуре данных
        - Все числовые поля ОБЯЗАТЕЛЬНО должны быть числами, не строками
        - Все обязательные поля должны присутствовать
        - НЕ используй запятые в конце объектов или массивов

        СТРОГО: отвечай только JSON, начинающийся с {{ и заканчивающийся }}"""
        last_error = None
        for attempt in range(max_attempts):
            try:
                messages = [
                    LLMMessage(role="user", content=f"{system_prompt}\n\nДанные для анализа:\n{data_to_analyze}")]
                response = await self.complete_chat(messages, temperature=0.05 if attempt == 0 else 0.1)
                parsed_result = self._ultra_robust_json_parser(response.content)
                validated_result = self._validate_and_fix_json_structure(parsed_result)
                return validated_result
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    await asyncio.sleep(1 + attempt)
                else:
                    return self._create_emergency_fallback_result(extraction_prompt, str(e))
        return self._create_emergency_fallback_result(extraction_prompt, str(last_error))

    async def close(self):
        pass


class GigaChatRiskAnalysisLLMClient(GigaChatLLMClient):
    """Специализированный GigaChat клиент для анализа рисков"""

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        if self.config.temperature > 0.3:
            self.config.temperature = 0.2

    async def evaluate_risk(
            self,
            risk_type: str,
            agent_data: str,
            evaluation_criteria: str,
            examples: Optional[str] = None
    ) -> Dict[str, Any]:
        print(f"🧠 РАССУЖДЕНИЯ АГЕНТА: Начинаю анализ {risk_type}")
        print(f"🧠 РАССУЖДЕНИЯ АГЕНТА: Изучаю данные агента...")
        system_prompt = f"""Ты эксперт по оценке рисков ИИ-агентов в банковской сфере. 

🎯 ЗАДАЧА: Оценить {risk_type} для предоставленного агента

📋 КРИТЕРИИ ОЦЕНКИ: {evaluation_criteria}

🧠 ВАЖНО: Сначала ПОДРОБНО рассуждай вслух:
1. Опиши что ты видишь в данных агента
2. Проанализируй какие факторы влияют на данный тип риска
3. Объясни почему выбираешь такую оценку вероятности (1-5)
4. Обоснуй почему выбираешь такую оценку воздействия (1-5)
5. Предложи конкретные рекомендации

После рассуждений дай ТОЛЬКО чистый JSON (без markdown):
{{
    "probability_score": число_от_1_до_5,
    "impact_score": число_от_1_до_5,
    "total_score": вероятность_умножить_на_воздействие,
    "risk_level": "low_или_medium_или_high",
    "probability_reasoning": "краткое_обоснование_вероятности",
    "impact_reasoning": "краткое_обоснование_воздействия",
    "key_factors": ["ключевой_фактор1", "ключевой_фактор2"],
    "recommendations": ["рекомендация1", "рекомендация2"],
    "confidence_level": число_от_0.0_до_1.0
}}"""
        try:
            print(f"🧠 РАССУЖДЕНИЯ АГЕНТА: Отправляю запрос с инструкциями...")
            loop = asyncio.get_event_loop()

            def sync_invoke():
                prompt = f"{system_prompt}\n\n📊 ДАННЫЕ АГЕНТА:\n{agent_data[:1500]}"
                return self.gigachat.invoke(prompt)

            response = await loop.run_in_executor(None, sync_invoke)
            raw_content = response.content if hasattr(response, 'content') else str(response)
            print(f"🧠 РАССУЖДЕНИЯ АГЕНТА: Получен ответ длиной {len(raw_content)} символов")
            reasoning_shown = False
            if len(raw_content) > 100:
                json_start = raw_content.find('{')
                if json_start > 100:
                    reasoning_text = raw_content[:json_start].strip()
                    json_part = raw_content[json_start:]
                    reasoning_text = reasoning_text.replace('```', '').replace('json', '').strip()
                    if reasoning_text and len(reasoning_text) > 50:
                        print(f"\n{'=' * 70}")
                        print(f"🧠 РАССУЖДЕНИЯ АГЕНТА ПО ТИПУ РИСКА: {risk_type.upper()}")
                        print(f"{'=' * 70}")
                        print(reasoning_text)
                        print(f"{'=' * 70}\n")
                        reasoning_shown = True
                    try:
                        parsed_data = self._parse_gigachat_response(json_part)
                    except:
                        parsed_data = self._parse_gigachat_response(raw_content)
                else:
                    parsed_data = self._parse_gigachat_response(raw_content)
            else:
                parsed_data = self._parse_gigachat_response(raw_content)
            if not reasoning_shown:
                print(f"🧠 РАССУЖДЕНИЯ АГЕНТА: ⚠️  Развернутые рассуждения не найдены в ответе")
            validated_data = self._validate_gigachat_response(parsed_data, risk_type)
            print(f"🧠 РАССУЖДЕНИЯ АГЕНТА: ✅ Анализ {risk_type} завершен")
            return validated_data
        except Exception as e:
            print(f"🧠 РАССУЖДЕНИЯ АГЕНТА: ❌ Ошибка - {e}")
            return self._create_fallback_response(risk_type, f"Ошибка GigaChat: {e}")

    def _parse_gigachat_response(self, content: str) -> Dict[str, Any]:
        import json
        import re
        content = content.strip()
        content = re.sub(r'^.*?({.*}).*$', r'\1', content, flags=re.DOTALL)
        if not content.startswith('{'):
            json_match = re.search(r'{[^{}]*(?:{[^{}]*}[^{}]*)*}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            else:
                raise ValueError(f"Не найден JSON в ответе: {content[:100]}")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"🔍 GIGACHAT DEBUG: Ошибка JSON парсинга: {e}")
            print(f"🔍 GIGACHAT DEBUG: Проблемный контент: {content}")
            fixed_content = self._fix_json_for_gigachat(content)
            return json.loads(fixed_content)

    def _fix_json_for_gigachat(self, content: str) -> str:
        import re
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        content = re.sub(r':\s*([^",{\[\]\s][^,}\]]*[^",}\]\s])\s*[,}]',
                         lambda m: f': "{m.group(1).strip()}"' + m.group(0)[-1], content)
        content = re.sub(r':\s*true', ': "true"', content)
        content = re.sub(r':\s*false', ': "false"', content)
        return content

    def _validate_gigachat_response(self, data: Dict[str, Any], risk_type: str) -> Dict[str, Any]:
        defaults = {
            "probability_score": 3,
            "impact_score": 3,
            "total_score": 9,
            "risk_level": "medium",
            "probability_reasoning": f"GigaChat не предоставил обоснование для вероятности {risk_type}",
            "impact_reasoning": f"GigaChat не предоставил обоснование для воздействия {risk_type}",
            "key_factors": [],
            "recommendations": [f"Провести дополнительный анализ {risk_type}", "Улучшить мониторинг"],
            "confidence_level": 0.7
        }
        for field, default in defaults.items():
            if field not in data or not data[field]:
                data[field] = default
                print(f"🔧 GIGACHAT: Поле {field} заменено на дефолт: {default}")
        if not data["key_factors"] or len(data["key_factors"]) == 0:
            factors = []
            prob_text = str(data.get("probability_reasoning", "")).lower()
            if "недостаточн" in prob_text and "защит" in prob_text:
                factors.append("Недостаточные меры защиты")
            if "guardrails" in prob_text:
                factors.append("Отсутствие guardrails")
            if "автоном" in prob_text:
                factors.append("Высокий уровень автономности")
            if "интеграц" in prob_text:
                factors.append("Интеграция с внешними API")
            if "данны" in prob_text and "персональн" in prob_text:
                factors.append("Обработка персональных данных")
            impact_text = str(data.get("impact_reasoning", "")).lower()
            if "репутац" in impact_text:
                factors.append("Репутационные риски")
            if "юридическ" in impact_text:
                factors.append("Юридические последствия")
            if "штраф" in impact_text:
                factors.append("Финансовые потери")
            if "доверие" in impact_text:
                factors.append("Потеря доверия пользователей")
            if factors:
                data["key_factors"] = factors[:5]
                print(f"🔧 GIGACHAT: Извлечены key_factors: {factors}")
            else:
                fallback_factors = {
                    "ethical": ["Потенциальная дискриминация", "Этические нарушения"],
                    "social": ["Манипуляция пользователями", "Распространение дезинформации"],
                    "security": ["Уязвимости безопасности", "Утечка данных"],
                    "stability": ["Нестабильность модели", "Ошибки в ответах"],
                    "autonomy": ["Неконтролируемые действия", "Превышение полномочий"],
                    "regulatory": ["Нарушение регуляторных требований", "Штрафные санкции"]
                }
                data["key_factors"] = fallback_factors.get(risk_type, ["Неопределенные факторы риска"])
        try:
            data["probability_score"] = max(1, min(5, int(float(str(data["probability_score"])))))
            data["impact_score"] = max(1, min(5, int(float(str(data["impact_score"])))))
            data["total_score"] = data["probability_score"] * data["impact_score"]
        except (ValueError, TypeError) as e:
            print(f"🔧 GIGACHAT: Ошибка валидации чисел: {e}")
            data["probability_score"] = 3
            data["impact_score"] = 3
            data["total_score"] = 9
        valid_levels = ["low", "medium", "high"]
        if data.get("risk_level") not in valid_levels:
            score = data["total_score"]
            if score <= 6:
                data["risk_level"] = "low"
            elif score <= 14:
                data["risk_level"] = "medium"
            else:
                data["risk_level"] = "high"
        if not isinstance(data.get("recommendations"), list):
            data["recommendations"] = [f"Улучшить анализ {risk_type}"]
        if not isinstance(data.get("key_factors"), list):
            data["key_factors"] = []
        return data

    def _create_fallback_response(self, risk_type: str, error_msg: str) -> Dict[str, Any]:
        return {
            "probability_score": 3,
            "impact_score": 3,
            "total_score": 9,
            "risk_level": "medium",
            "probability_reasoning": f"Fallback оценка для {risk_type}: {error_msg}",
            "impact_reasoning": f"Fallback оценка для {risk_type}: {error_msg}",
            "key_factors": ["Ошибка получения данных от GigaChat"],
            "recommendations": [f"Повторить оценку {risk_type}", "Проверить подключение к GigaChat"],
            "confidence_level": 0.3
        }

    async def critique_evaluation(
            self,
            risk_type: str,
            original_evaluation: Dict[str, Any],
            agent_data: str,
            quality_threshold: float = 7.0
    ) -> Dict[str, Any]:
        system_prompt = f"""Ты - критик-эксперт по оценке качества анализа рисков ИИ-агентов.

Твоя задача: оценить качество предоставленной оценки {risk_type}.

КРИТЕРИИ КАЧЕСТВА:
1. Обоснованность оценок (соответствие данным агента)
2. Полнота анализа (учтены ли все аспекты)
3. Логичность рассуждений
4. Практичность рекомендаций
5. Соответствие методике оценки

ШКАЛА КАЧЕСТВА: 0-10 баллов
ПОРОГ ПРИЕМЛЕМОСТИ: {quality_threshold} баллов

ФОРМАТ ОТВЕТА (СТРОГО JSON):
{{
    "quality_score": <0.0-10.0>,
    "is_acceptable": <true|false>,
    "issues_found": ["<проблема1>", "<проблема2>", ...],
    "improvement_suggestions": ["<предложение1>", "<предложение2>", ...],
    "critic_reasoning": "<подробное обоснование оценки качества>"
}}"""
        evaluation_text = json.dumps(original_evaluation, ensure_ascii=False, indent=2)
        context = f"""ДАННЫЕ ОБ АГЕНТЕ:
{agent_data}

ОЦЕНКА ДЛЯ КРИТИКИ:
{evaluation_text}"""
        response = await self.extract_structured_data(
            data_to_analyze=context,
            extraction_prompt="Критически оцени качество представленной оценки риска",
            expected_format="JSON"
        )
        if "quality_score" not in response:
            response["quality_score"] = 7.0
        if "is_acceptable" not in response:
            response["is_acceptable"] = response["quality_score"] >= quality_threshold
        if "critic_reasoning" not in response:
            response["critic_reasoning"] = "Автоматически сгенерированная оценка"
        return response


class RiskAnalysisLLMClient(LLMClient):
    """Специализированный клиент для анализа рисков"""

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        if self.config.temperature > 0.3:
            self.config.temperature = 0.2

    async def evaluate_risk(
            self,
            risk_type: str,
            agent_data: str,
            evaluation_criteria: str,
            examples: Optional[str] = None
    ) -> Dict[str, Any]:
        system_prompt = f"""Ты - эксперт по оценке операционных рисков ИИ-агентов в банковской сфере.

Твоя задача: оценить {risk_type} для предоставленного ИИ-агента.

КРИТЕРИИ ОЦЕНКИ:
{evaluation_criteria}

ШКАЛА ОЦЕНКИ:
- Вероятность: 1-5 баллов (1=низкая, 5=высокая)
- Тяжесть: 1-5 баллов (1=низкие потери, 5=высокие потери)
- Итоговый балл = Вероятность × Тяжесть

ФОРМАТ ОТВЕТА (СТРОГО JSON):
{{
    "probability_score": <1-5>,
    "impact_score": <1-5>,
    "total_score": <1-25>,
    "risk_level": "<low|medium|high>",
    "probability_reasoning": "<подробное обоснование вероятности>",
    "impact_reasoning": "<подробное обоснование тяжести>",
    "key_factors": ["<фактор1>", "<фактор2>", ...],
    "recommendations": ["<рекомендация1>", "<рекомендация2>", ...],
    "confidence_level": <0.0-1.0>
}}

УРОВНИ РИСКА:
- low: 1-6 баллов
- medium: 7-14 баллов  
- high: 15-25 баллов"""
        if examples:
            system_prompt += f"\n\nПРИМЕРЫ ОЦЕНОК:\n{examples}"
        response = await self.extract_structured_data(
            data_to_analyze=agent_data,
            extraction_prompt=f"Оцени {risk_type} согласно методике",
            expected_format="JSON"
        )
        required_fields = [
            "probability_score", "impact_score", "total_score",
            "risk_level", "probability_reasoning", "impact_reasoning"
        ]
        for field in required_fields:
            if field not in response:
                raise LLMError(f"Отсутствует обязательное поле в ответе: {field}")
        if not (1 <= response["probability_score"] <= 5):
            raise LLMError(f"Некорректный probability_score: {response['probability_score']}")
        if not (1 <= response["impact_score"] <= 5):
            raise LLMError(f"Некорректный impact_score: {response['impact_score']}")
        response["total_score"] = response["probability_score"] * response["impact_score"]
        score = response["total_score"]
        if score <= 6:
            response["risk_level"] = "low"
        elif score <= 14:
            response["risk_level"] = "medium"
        else:
            response["risk_level"] = "high"
        return response

    async def critique_evaluation(
            self,
            risk_type: str,
            original_evaluation: Dict[str, Any],
            agent_data: str,
            quality_threshold: float = 7.0
    ) -> Dict[str, Any]:
        system_prompt = f"""Ты - критик-эксперт по оценке качества анализа рисков ИИ-агентов.

Твоя задача: оценить качество предоставленной оценки {risk_type}.

КРИТЕРИИ КАЧЕСТВА:
1. Обоснованность оценок (соответствие данным агента)
2. Полнота анализа (учтены ли все аспекты)
3. Логичность рассуждений
4. Практичность рекомендаций
5. Соответствие методике оценки

ШКАЛА КАЧЕСТВА: 0-10 баллов
ПОРОГ ПРИЕМЛЕМОСТИ: {quality_threshold} баллов

ФОРМАТ ОТВЕТА (СТРОГО JSON):
{{
    "quality_score": <0.0-10.0>,
    "is_acceptable": <true|false>,
    "issues_found": ["<проблема1>", "<проблема2>", ...],
    "improvement_suggestions": ["<предложение1>", "<предложение2>", ...],
    "critic_reasoning": "<подробное обоснование оценки качества>"
}}"""
        evaluation_text = json.dumps(original_evaluation, ensure_ascii=False, indent=2)
        context = f"""ДАННЫЕ ОБ АГЕНТЕ:
{agent_data}

ОЦЕНКА ДЛЯ КРИТИКИ:
{evaluation_text}"""
        response = await self.extract_structured_data(
            data_to_analyze=context,
            extraction_prompt="Критически оцени качество представленной оценки риска",
            expected_format="JSON"
        )
        required_fields = ["quality_score", "is_acceptable", "critic_reasoning"]
        for field in required_fields:
            if field not in response:
                raise LLMError(f"Отсутствует обязательное поле: {field}")
        response["is_acceptable"] = response["quality_score"] >= quality_threshold
        return response


def create_llm_client(
        client_type: str = "standard",
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
) -> LLMClient:
    """
    Фабрика для создания LLM клиентов
    """
    overrides = {}
    if base_url is not None:
        overrides['base_url'] = base_url
    if model is not None:
        overrides['model'] = model
    if temperature is not None:
        overrides['temperature'] = temperature
    config = LLMConfig.from_manager(**overrides)
    if config.provider == LLMProvider.GIGACHAT:
        if client_type == "risk_analysis":
            return GigaChatRiskAnalysisLLMClient(config)
        else:
            return GigaChatLLMClient(config)
    elif config.provider == LLMProvider.DEEPSEEK:
        if client_type == "risk_analysis":
            return DeepSeekRiskAnalysisLLMClient(config)
        else:
            return DeepSeekLLMClient(config)
    else:
        if client_type == "risk_analysis":
            return RiskAnalysisLLMClient(config)
        else:
            return LLMClient(config)


_global_client: Optional[LLMClient] = None


async def get_llm_client() -> LLMClient:
    """
    Получение глобального LLM клиента
    """
    global _global_client
    if _global_client is None:
        try:
            config = LLMConfig.from_manager()
            print(f"🔧 Загружена конфигурация LLM:")
            print(f"   Провайдер: {config.provider.value}")
            print(f"   URL: {config.base_url}")
            print(f"   Модель: {config.model}")
            if config.provider == LLMProvider.GIGACHAT:
                print("🤖 Создаем GigaChat клиент...")
                _global_client = GigaChatLLMClient(config)
            elif config.provider == LLMProvider.DEEPSEEK:
                print("🤖 Создаем DeepSeek клиент...")
                _global_client = DeepSeekLLMClient(config)
            else:
                print(f"🤖 Создаем {config.provider.value} клиент...")
                _global_client = LLMClient(config)
            print("🔍 Проверяем доступность LLM сервера...")
            is_available = await _global_client.health_check()
            if not is_available:
                provider_name = config.provider.value
                error_msg = f"{provider_name} сервер недоступен. Проверьте настройки подключения."
                if config.provider == LLMProvider.GIGACHAT:
                    error_msg += f"\nПроверьте:\n- Сертификаты: {config.cert_file}, {config.key_file}\n- URL: {config.base_url}"
                elif config.provider == LLMProvider.DEEPSEEK:
                    error_msg += f"\nПроверьте:\n- API ключ\n- URL: {config.base_url}"
                else:
                    error_msg += f"\nПроверьте:\n- URL: {config.base_url}\n- Запущен ли сервер?"
                raise LLMError(error_msg)
            print(f"✅ {config.provider.value} клиент успешно создан и проверен")
        except Exception as e:
            print("❌ ОШИБКА СОЗДАНИЯ LLM КЛИЕНТА:")
            print(f"   {str(e)}")
            print("\n🔍 Запускаем диагностику...")
            print_llm_diagnosis()
            raise e
    return _global_client


def reset_global_client():
    global _global_client
    _global_client = None


def force_recreate_global_client():
    global _global_client
    _global_client = None


def diagnose_llm_configuration() -> Dict[str, Any]:
    import os
    diagnosis = {
        "environment_variables": {},
        "config_manager_info": {},
        "files_exist": {},
        "errors": []
    }
    env_vars = [
        "LLM_PROVIDER", "GIGACHAT_BASE_URL", "GIGACHAT_MODEL",
        "GIGACHAT_CERT_PATH", "GIGACHAT_KEY_PATH", "LLM_DeepSeek_API_KEY"
    ]
    for var in env_vars:
        diagnosis["environment_variables"][var] = os.getenv(var, "НЕ УСТАНОВЛЕНА")
    try:
        manager = get_llm_config_manager()
        diagnosis["config_manager_info"] = manager.get_info()
    except Exception as e:
        diagnosis["errors"].append(f"Ошибка менеджера конфигурации: {str(e)}")
    if os.getenv("LLM_PROVIDER", "").lower() == "gigachat":
        cert_path = os.getenv("GIGACHAT_CERT_PATH", "")
        key_path = os.getenv("GIGACHAT_KEY_PATH", "")
        if cert_path:
            if not os.path.isabs(cert_path):
                cert_path = os.path.join(os.getcwd(), cert_path)
            diagnosis["files_exist"]["cert_file"] = os.path.exists(cert_path)
            diagnosis["files_exist"]["cert_path"] = cert_path
        if key_path:
            if not os.path.isabs(key_path):
                key_path = os.path.join(os.getcwd(), key_path)
            diagnosis["files_exist"]["key_file"] = os.path.exists(key_path)
            diagnosis["files_exist"]["key_path"] = key_path
    return diagnosis


def print_llm_diagnosis():
    import json
    diagnosis = diagnose_llm_configuration()
    print("🔍 ДИАГНОСТИКА КОНФИГУРАЦИИ LLM:")
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))


async def test_gigachat_direct() -> Dict[str, Any]:
    print("🧪 ПРЯМОЕ ТЕСТИРОВАНИЕ GIGACHAT")
    print("=" * 50)
    result = {
        "success": False,
        "error": None,
        "response": None,
        "config_info": {},
        "certificate_check": {}
    }
    try:
        from .llm_config_manager import get_llm_config_manager
        manager = get_llm_config_manager()
        config_info = manager.get_info()
        result["config_info"] = config_info
        print(f"📋 Конфигурация:")
        print(f"   Provider: {config_info['provider']}")
        print(f"   URL: {config_info['base_url']}")
        print(f"   Model: {config_info['model']}")
        print(f"   Cert: {config_info.get('cert_file', 'не указан')}")
        print(f"   Key: {config_info.get('key_file', 'не указан')}")
        import os
        cert_exists = os.path.exists(config_info.get('cert_file', ''))
        key_exists = os.path.exists(config_info.get('key_file', ''))
        result["certificate_check"] = {
            "cert_exists": cert_exists,
            "key_exists": key_exists,
            "cert_path": config_info.get('cert_file'),
            "key_path": config_info.get('key_file')
        }
        print(f"🔒 Проверка сертификатов:")
        print(f"   Cert файл: {'✅' if cert_exists else '❌'}")
        print(f"   Key файл: {'✅' if key_exists else '❌'}")
        if not (cert_exists and key_exists):
            result["error"] = "Сертификаты не найдены"
            return result
        if not GIGACHAT_AVAILABLE:
            result["error"] = "langchain_gigachat не установлен"
            return result
        gigachat = GigaChat(
            base_url=config_info['base_url'],
            cert_file=config_info['cert_file'],
            key_file=config_info['key_file'],
            model=config_info['model'],
            temperature=config_info['temperature'],
            top_p=config_info.get('top_p', 0.2),
            verify_ssl_certs=config_info.get('verify_ssl_certs', False),
            profanity_check=config_info.get('profanity_check', False),
            streaming=config_info.get('streaming', True)
        )
        print("✅ GigaChat клиент создан")
        print("📞 Тестируем вызов GigaChat...")
        import asyncio
        loop = asyncio.get_event_loop()

        def sync_call():
            return gigachat.invoke("Привет! Ответь кратко.")

        response = await loop.run_in_executor(None, sync_call)
        print(f"📨 Получен ответ: {type(response)}")
        if hasattr(response, 'content'):
            content = response.content
            print(f"📝 Содержимое: '{content}'")
            result["response"] = {
                "type": str(type(response)),
                "content": content,
                "has_content": True,
                "content_length": len(content) if content else 0
            }
        else:
            print(f"⚠️ Ответ без атрибута content: {response}")
            result["response"] = {
                "type": str(type(response)),
                "content": str(response),
                "has_content": False,
                "raw_response": str(response)
            }
        result["success"] = True
        print("🎉 ТЕСТ ПРОШЕЛ УСПЕШНО!")
    except Exception as e:
        result["error"] = str(e)
        result["exception_type"] = type(e).__name__
        print(f"❌ ОШИБКА: {e}")
        print(f"❌ Тип ошибки: {type(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
    return result


async def test_deepseek_direct() -> Dict[str, Any]:
    """Прямое тестирование DeepSeek для диагностики"""
    print("🧪 ПРЯМОЕ ТЕСТИРОВАНИЕ DEEPSEEK")
    print("=" * 50)
    result = {
        "success": False,
        "error": None,
        "response": None,
        "config_info": {}
    }
    try:
        from .llm_config_manager import get_llm_config_manager
        manager = get_llm_config_manager()
        config_info = manager.get_info()
        result["config_info"] = config_info
        print(f"📋 Конфигурация:")
        print(f"   Provider: {config_info['provider']}")
        print(f"   URL: {config_info['base_url']}")
        print(f"   Model: {config_info['model']}")
        print(f"   API Key: {'******' if config_info.get('api_key') else 'не указан'}")
        if not DEEPSEEK_AVAILABLE:
            result["error"] = "openai не установлен"
            return result
        if not config_info.get('api_key'):
            result["error"] = "API ключ для DeepSeek не указан"
            return result
        client = AsyncOpenAI(
            api_key=config_info['api_key'],
            base_url=config_info['base_url']
        )
        print("✅ DeepSeek клиент создан")
        print("📞 Тестируем вызов DeepSeek...")
        response = await client.chat.completions.create(
            model=config_info['model'],
            messages=[{"role": "user", "content": "Привет! Ответь кратко."}],
            temperature=config_info['temperature'],
            max_tokens=50
        )
        content = response.choices[0].message.content
        print(f"📝 Содержимое: '{content}'")
        result["response"] = {
            "type": str(type(response)),
            "content": content,
            "has_content": True,
            "content_length": len(content) if content else 0
        }
        result["success"] = True
        print("🎉 ТЕСТ ПРОШЕЛ УСПЕШНО!")
    except Exception as e:
        result["error"] = str(e)
        result["exception_type"] = type(e).__name__
        print(f"❌ ОШИБКА: {e}")
        print(f"❌ Тип ошибки: {type(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
    return result


__all__ = [
    "LLMClient",
    "GigaChatLLMClient",
    "GigaChatRiskAnalysisLLMClient",
    "DeepSeekLLMClient",
    "DeepSeekRiskAnalysisLLMClient",
    "RiskAnalysisLLMClient",
    "LLMConfig",
    "LLMMessage",
    "LLMResponse",
    "LLMError",
    "create_llm_client",
    "get_llm_client",
    "reset_global_client",
    "force_recreate_global_client",
    "diagnose_llm_configuration",
    "print_llm_diagnosis",
    "test_gigachat_direct",
    "test_deepseek_direct"
]