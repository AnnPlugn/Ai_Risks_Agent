# src/models/risk_models.py - ПОЛНОСТЬЮ ОБНОВЛЕННАЯ ВЕРСИЯ
"""
Модели данных для системы оценки рисков ИИ-агентов
Обновлено для исправления ошибок WorkflowState
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Annotated
from enum import Enum

from pydantic import BaseModel, Field, validator


# ===============================
# Перечисления (Enums)
# ===============================

class RiskType(str, Enum):
    """Типы рисков ИИ-агентов"""
    ETHICAL = "ethical"           # Этические и дискриминационные риски
    STABILITY = "stability"       # Риски ошибок и нестабильности LLM
    SECURITY = "security"         # Риски безопасности данных и систем  
    AUTONOMY = "autonomy"         # Риски автономности и управления
    REGULATORY = "regulatory"     # Регуляторные и юридические риски
    SOCIAL = "social"            # Социальные и манипулятивные риски


class RiskLevel(str, Enum):
    """Уровни риска"""
    LOW = "low"           # 1-6 баллов
    MEDIUM = "medium"     # 7-14 баллов  
    HIGH = "high"         # 15-25 баллов


class AgentType(str, Enum):
    """Типы ИИ-агентов"""
    CHATBOT = "chatbot"
    ASSISTANT = "assistant"
    ANALYZER = "analyzer" 
    ADVISOR = "advisor"
    MODERATOR = "moderator"
    GENERATOR = "generator"
    OTHER = "other"


class AutonomyLevel(str, Enum):
    """Уровни автономности агента"""
    SUPERVISED = "supervised"         # Под постоянным надзором
    SEMI_AUTONOMOUS = "semi_autonomous"  # Частично автономный
    AUTONOMOUS = "autonomous"         # Полностью автономный


class DataSensitivity(str, Enum):
    """Уровни чувствительности данных"""
    PUBLIC = "public"           # Открытые данные
    INTERNAL = "internal"       # Внутренние данные компании
    CONFIDENTIAL = "confidential"  # Конфиденциальные данные
    PERSONAL = "personal"       # Персональные данные
    FINANCIAL = "financial"     # Финансовые данные


class ProcessingStatus(str, Enum):
    """Статусы обработки"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY_NEEDED = "retry_needed"


# ===============================
# Основные модели данных
# ===============================

class AgentProfile(BaseModel):
    """ИСПРАВЛЕННЫЙ профиль ИИ-агента с полем external_apis"""
    
    # Основная информация
    name: str = Field(..., description="Название агента")
    description: str = Field(..., description="Описание функций агента")
    version: Optional[str] = Field(default="1.0", description="Версия агента")
    
    # Классификация
    agent_type: AgentType = Field(..., description="Тип агента")
    llm_model: str = Field(..., description="Используемая LLM модель")
    autonomy_level: AutonomyLevel = Field(..., description="Уровень автономности")
    
    # Доступ к данным и API
    data_access: List[DataSensitivity] = Field(default_factory=list, description="Типы доступных данных")
    external_apis: List[str] = Field(default_factory=list, description="Внешние API, к которым есть доступ")
    
    # Бизнес-контекст
    target_audience: str = Field(..., description="Целевая аудитория")
    operations_per_hour: Optional[int] = Field(default=None, description="Операций в час")
    revenue_per_operation: Optional[float] = Field(default=None, description="Доход с операции")
    
    # Техническая документация
    system_prompts: List[str] = Field(default_factory=list, description="Системные промпты")
    guardrails: List[str] = Field(default_factory=list, description="Ограничения безопасности")
    source_files: List[str] = Field(default_factory=list, description="Исходные файлы анализа")
    detailed_summary: Optional[Dict[str, str]] = Field(
        None, 
        description="Детальное саммари агента с разделами анализа"
    )
    # Метаданные
    created_at: Optional[datetime] = Field(default=None, description="Время создания")
    updated_at: Optional[datetime] = Field(default=None, description="Время обновления")
    
    @validator('version', pre=True, always=True)
    def set_default_version(cls, v):
        return v or "1.0"
    
    @validator('created_at', 'updated_at', pre=True, always=True)
    def set_timestamps(cls, v):
        if v is None:
            return datetime.now()
        return v

class ThreatAssessment(BaseModel):
    """Модель для оценки отдельной угрозы"""
    risk_level: str = Field(..., description="Уровень риска угрозы: низкая, средняя, высокая")
    probability_score: int = Field(..., ge=1, le=5, description="Вероятность угрозы (1-5)")
    impact_score: int = Field(..., ge=1, le=5, description="Воздействие угрозы (1-5)")
    reasoning: str = Field(..., description="Детальное обоснование оценки угрозы")

# src/models/risk_models.py - КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ RiskEvaluation

class RiskEvaluation(BaseModel):
    """ПОЛНОСТЬЮ ИСПРАВЛЕННАЯ оценка риска с правильными полями"""
    
    # Идентификация
    risk_type: RiskType = Field(..., description="Тип риска")
    evaluator_agent: str = Field(..., description="Агент, проводивший оценку")
    
    # Основные оценки
    probability_score: int = Field(..., ge=1, le=5, description="Вероятность риска (1-5)")
    impact_score: int = Field(..., ge=1, le=5, description="Тяжесть последствий (1-5)")
    total_score: int = Field(..., description="Общий балл риска")
    risk_level: RiskLevel = Field(..., description="Уровень риска")
    
    # Обоснования
    probability_reasoning: str = Field(..., description="Обоснование вероятности")
    impact_reasoning: str = Field(..., description="Обоснование тяжести")
    
    # ИСПРАВЛЕНО: Используем key_factors вместо identified_risks (как в БД)
    key_factors: List[str] = Field(default_factory=list, description="Ключевые факторы риска")
    recommendations: List[str] = Field(default_factory=list, description="Рекомендации")
    
    # Метаданные
    confidence_level: float = Field(default=0.7, ge=0.0, le=1.0, description="Уровень уверенности")
    timestamp: datetime = Field(default_factory=datetime.now)
    # НОВОЕ ПОЛЕ ДЛЯ ДЕТАЛЬНЫХ УГРОЗ
    threat_assessments: Optional[Dict[str, ThreatAssessment]] = Field(
        default=None, 
        description="Детальная оценка отдельных угроз (для EthicalRiskEvaluator)"
    )
    def __init__(self, **data):
        # Автовычисление total_score если не задан
        if 'total_score' not in data or data['total_score'] is None:
            prob = data.get('probability_score', 3)
            impact = data.get('impact_score', 3)
            data['total_score'] = prob * impact
        
        # Автовычисление risk_level если не задан
        if 'risk_level' not in data or data['risk_level'] is None:
            total = data.get('total_score', 9)
            if total <= 6:
                data['risk_level'] = RiskLevel.LOW
            elif total <= 14:
                data['risk_level'] = RiskLevel.MEDIUM
            else:
                data['risk_level'] = RiskLevel.HIGH
        
        super().__init__(**data)
    
    @classmethod
    def create_safe(
        cls,
        risk_type: RiskType,
        evaluator_agent: str,
        raw_data: Dict[str, Any]
    ) -> 'RiskEvaluation':
        """ОБНОВЛЕННОЕ безопасное создание RiskEvaluation с поддержкой threat_assessments"""
        
        try:
            # Извлекаем основные поля (существующий код)
            probability_score = int(raw_data.get("probability_score", 3))
            impact_score = int(raw_data.get("impact_score", 3))
            total_score = int(raw_data.get("total_score", probability_score * impact_score))
            
            # Определяем уровень риска
            if total_score <= 6:
                risk_level = RiskLevel.LOW
            elif total_score <= 14:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.HIGH
            
            # НОВАЯ ЛОГИКА: Обработка threat_assessments
            threat_assessments = None
            if "threat_assessments" in raw_data:
                threat_data = raw_data["threat_assessments"]
                if isinstance(threat_data, dict):
                    threat_assessments = {}
                    for threat_name, threat_info in threat_data.items():
                        if isinstance(threat_info, dict):
                            try:
                                threat_assessments[threat_name] = ThreatAssessment(
                                    risk_level=threat_info.get("risk_level", "средняя"),
                                    probability_score=int(threat_info.get("probability_score", 3)),
                                    impact_score=int(threat_info.get("impact_score", 3)),
                                    reasoning=str(threat_info.get("reasoning", "Обоснование отсутствует"))
                                )
                            except Exception as e:
                                print(f"⚠️ Ошибка парсинга угрозы {threat_name}: {e}")
                                # Создаем fallback для конкретной угрозы
                                threat_assessments[threat_name] = ThreatAssessment(
                                    risk_level="средняя",
                                    probability_score=3,
                                    impact_score=3,
                                    reasoning=f"Fallback для угрозы {threat_name} из-за ошибки парсинга"
                                )
            
            # Создаем объект с новым полем
            return cls(
                risk_type=risk_type,
                evaluator_agent=evaluator_agent,
                probability_score=probability_score,
                impact_score=impact_score,
                total_score=total_score,
                risk_level=risk_level,
                probability_reasoning=raw_data.get("probability_reasoning", "Обоснование отсутствует"),
                impact_reasoning=raw_data.get("impact_reasoning", "Обоснование отсутствует"),
                key_factors=raw_data.get("key_factors", []),
                recommendations=raw_data.get("recommendations", []),
                confidence_level=float(raw_data.get("confidence_level", 0.7)),
                timestamp=datetime.now(),
                threat_assessments=threat_assessments  # НОВОЕ ПОЛЕ
            )
            
        except Exception as e:
            # Логируем ошибку и создаем fallback объект
            print(f"❌ Ошибка создания RiskEvaluation: {e}")
            
            return cls._create_fallback(risk_type, evaluator_agent, raw_data)
    
    @classmethod
    def _create_fallback(
        cls,
        risk_type: RiskType,
        evaluator_agent: str,
        raw_data: Dict[str, Any]
    ) -> 'RiskEvaluation':
        """Создание fallback объекта при ошибках"""
        
        # Создаем fallback threat_assessments для EthicalRiskEvaluator
        threat_assessments = None
        if evaluator_agent == "EthicalRiskEvaluator":
            threat_assessments = {
                "галлюцинации_и_зацикливание": ThreatAssessment(
                    risk_level="средняя",
                    probability_score=3,
                    impact_score=3,
                    reasoning="Fallback оценка: средний риск галлюцинаций из-за отсутствия данных"
                ),
                "дезинформация": ThreatAssessment(
                    risk_level="средняя",
                    probability_score=3,
                    impact_score=3,
                    reasoning="Fallback оценка: средний риск дезинформации из-за отсутствия данных"
                ),
                "токсичность_и_дискриминация": ThreatAssessment(
                    risk_level="средняя",
                    probability_score=3,
                    impact_score=3,
                    reasoning="Fallback оценка: средний риск токсичности из-за отсутствия данных"
                )
            }
        
        return cls(
            risk_type=risk_type,
            evaluator_agent=evaluator_agent,
            probability_score=3,
            impact_score=3,
            total_score=9,
            risk_level=RiskLevel.MEDIUM,
            probability_reasoning=f"Fallback для {evaluator_agent}",
            impact_reasoning=f"Fallback для {evaluator_agent}",
            key_factors=["Ошибка получения данных"],
            recommendations=["Повторить оценку"],
            confidence_level=0.3,
            timestamp=datetime.now(),
            threat_assessments=threat_assessments
        )

    @staticmethod
    def _safe_extract_int(value: Any, default: int, min_val: int, max_val: int) -> int:
        """Безопасное извлечение int"""
        try:
            val = int(float(value))  # Через float для обработки "3.0"
            return max(min_val, min(max_val, val))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _safe_extract_float(value: Any, default: float, min_val: float, max_val: float) -> float:
        """Безопасное извлечение float"""
        try:
            val = float(value)
            return max(min_val, min(max_val, val))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def _safe_extract_string(value: Any, default: str) -> str:
        """Безопасное извлечение строки"""
        if not value or not isinstance(value, str) or len(str(value).strip()) < 5:
            return default
        return str(value).strip()
    
    @staticmethod
    def _safe_extract_list(value: Any) -> List[str]:
        """Безопасное извлечение списка"""
        if not isinstance(value, list):
            return []
        
        result = []
        for item in value:
            if item and isinstance(item, str) and len(str(item).strip()) > 0:
                result.append(str(item).strip())
        
        return result[:10]  # Ограничиваем до 10


class CriticEvaluation(BaseModel):
    """Критическая оценка качества анализа риска"""
    
    # Оценка качества
    quality_score: float = Field(..., ge=0.0, le=10.0, description="Оценка качества (0-10)")
    is_acceptable: bool = Field(..., description="Приемлемо ли качество")
    
    # Анализ
    clarity_score: float = Field(..., ge=0.0, le=10.0, description="Ясность рассуждений")
    completeness_score: float = Field(..., ge=0.0, le=10.0, description="Полнота анализа")
    reasoning_score: float = Field(..., ge=0.0, le=10.0, description="Качество обоснований")
    
    # Комментарии
    feedback: str = Field(..., description="Обратная связь от критика")
    improvement_suggestions: List[str] = Field(default_factory=list, description="Предложения по улучшению")
    
    # Метаданные
    critic_agent: str = Field(..., description="Агент-критик")
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentRiskAssessment(BaseModel):
    """ИСПРАВЛЕННАЯ полная оценка рисков ИИ-агента"""
    
    # Идентификация
    assessment_id: str = Field(..., description="Уникальный ID оценки")
    agent_profile: AgentProfile = Field(..., description="Профиль анализируемого агента")
    
    # ИСПРАВЛЕНО: правильные типы полей
    risk_evaluations: Dict[str, RiskEvaluation] = Field(default_factory=dict, description="Оценки по типам рисков")
    critic_evaluations: Dict[str, CriticEvaluation] = Field(default_factory=dict, description="Критические оценки")
    
    # Общие результаты
    overall_risk_score: int = Field(..., description="Общий балл риска")
    overall_risk_level: RiskLevel = Field(..., description="Общий уровень риска")
    highest_risk_areas: List[str] = Field(default_factory=list, description="Области наивысшего риска (строки)")
    
    # Рекомендации
    priority_recommendations: List[str] = Field(default_factory=list, description="Приоритетные рекомендации")
    suggested_guardrails: List[str] = Field(default_factory=list, description="Предлагаемые ограничения")
    
    # Финансовые оценки
    total_expected_loss: Optional[float] = Field(default=None, description="Ожидаемые потери")
    
    # Обязательные поля процесса
    processing_time_seconds: float = Field(..., description="Время обработки в секундах")
    quality_checks_passed: bool = Field(..., description="Прошли ли проверки качества")
    
    # Метаданные процесса
    assessment_date: datetime = Field(default_factory=datetime.now, description="Дата оценки")
    processing_status: str = Field(default="completed", description="Статус обработки")
    
    @validator('risk_evaluations', pre=True)
    def validate_risk_evaluations(cls, v):
        """Валидация и преобразование risk_evaluations"""
        if isinstance(v, dict):
            result = {}
            for key, value in v.items():
                # Преобразуем enum ключи в строки
                if isinstance(key, RiskType):
                    key = key.value
                elif hasattr(key, 'value'):
                    key = key.value
                
                # Убеждаемся что value это RiskEvaluation
                if isinstance(value, dict):
                    result[str(key)] = RiskEvaluation(**value)
                elif isinstance(value, RiskEvaluation):
                    result[str(key)] = value
                else:
                    # Fallback
                    result[str(key)] = RiskEvaluation(
                        risk_type=RiskType(key) if key in [rt.value for rt in RiskType] else RiskType.ETHICAL,
                        evaluator_agent="unknown",
                        probability_score=3,
                        impact_score=3,
                        total_score=9,
                        risk_level=RiskLevel.MEDIUM,
                        probability_reasoning="Обоснование недоступно",
                        impact_reasoning="Обоснование недоступно"
                    )
            return result
        return v or {}
    
    @validator('highest_risk_areas', pre=True)
    def validate_highest_risk_areas(cls, v):
        """Преобразование highest_risk_areas в строки"""
        if not v:
            return []
        
        result = []
        for area in v:
            if isinstance(area, RiskType):
                result.append(area.value)
            elif hasattr(area, 'value'):
                result.append(area.value)
            else:
                result.append(str(area))
        return result
    
    @validator('overall_risk_level', pre=True)
    def validate_risk_level(cls, v):
        """Валидация уровня риска"""
        if isinstance(v, str):
            return RiskLevel(v)
        elif isinstance(v, RiskLevel):
            return v
        else:
            return RiskLevel.MEDIUM
    
    # Методы для совместимости
    def model_dump(self) -> Dict[str, Any]:
        """Замена устаревшего dict() метода"""
        return super().model_dump()
    
    def dict(self) -> Dict[str, Any]:
        """Обратная совместимость"""
        return self.model_dump()


# Вспомогательная функция для создания
def create_agent_risk_assessment(
    assessment_id: str,
    agent_profile: AgentProfile,
    risk_evaluations: Dict[str, RiskEvaluation],
    processing_time_seconds: float = 0.0,
    quality_checks_passed: bool = True
) -> AgentRiskAssessment:
    """Создание AgentRiskAssessment с правильной валидацией"""
    
    # Вычисляем общий балл риска
    if risk_evaluations:
        overall_score = max(eval.total_score for eval in risk_evaluations.values())
    else:
        overall_score = 0
    
    # Определяем уровень риска
    if overall_score <= 6:
        overall_level = RiskLevel.LOW
    elif overall_score <= 14:
        overall_level = RiskLevel.MEDIUM
    else:
        overall_level = RiskLevel.HIGH
    
    # Определяем области наивысшего риска (как строки)
    highest_risk_areas = []
    if risk_evaluations:
        sorted_risks = sorted(
            risk_evaluations.items(),
            key=lambda x: x[1].total_score,
            reverse=True
        )
        highest_risk_areas = [risk_type for risk_type, _ in sorted_risks[:3]]
    
    # Собираем рекомендации
    priority_recommendations = []
    for evaluation in risk_evaluations.values():
        priority_recommendations.extend(evaluation.recommendations[:2])
    
    return AgentRiskAssessment(
        assessment_id=assessment_id,
        agent_profile=agent_profile,
        risk_evaluations=risk_evaluations,
        overall_risk_score=overall_score,
        overall_risk_level=overall_level,
        highest_risk_areas=highest_risk_areas,
        priority_recommendations=priority_recommendations[:5],
        processing_time_seconds=processing_time_seconds,
        quality_checks_passed=quality_checks_passed
    )

class AgentTaskResult(BaseModel):
    """Результат работы отдельного агента"""
    
    agent_name: str = Field(..., description="Имя агента")
    task_type: str = Field(..., description="Тип задачи")
    status: ProcessingStatus = Field(..., description="Статус выполнения")
    
    # Результаты
    result_data: Optional[Dict[str, Any]] = Field(None, description="Данные результата")
    error_message: Optional[str] = Field(None, description="Сообщение об ошибке")
    
    # Метрики
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = Field(None)
    execution_time_seconds: Optional[float] = Field(None)
    assessment_id: Optional[str] = Field (None, description ="ID оценки")
    
    @validator('execution_time_seconds', always=True)
    def calculate_execution_time(cls, v, values):
        """Автоматический расчет времени выполнения"""
        if values.get('end_time') and values.get('start_time'):
            delta = values['end_time'] - values['start_time']
            return delta.total_seconds()
        return v


# ===============================
# WorkflowState - ИСПРАВЛЕННАЯ ВЕРСИЯ
# ===============================

class WorkflowState(BaseModel):
    """Состояние workflow для LangGraph - с поддержкой словарных методов"""
    
    # Идентификаторы
    assessment_id: Optional[str] = Field(None, description="ID оценки")
    preliminary_agent_name: Optional[str] = Field(None, description="Предварительное имя агента")
    
    # Входные данные
    source_files: List[str] = Field(default_factory=list, description="Файлы для анализа")
    agent_profile: Optional[Dict[str, Any]] = Field(None, description="Профиль агента с детальным саммари")
    
    # Промежуточные результаты
    profiling_result: Optional[Dict[str, Any]] = Field(None, description="Результат профилирования")
    
    # ИСПРАВЛЕНИЕ: Отдельные поля для каждого типа риска вместо общего словаря
    ethical_evaluation: Optional[Dict[str, Any]] = Field(None, description="Оценка этических рисков")
    stability_evaluation: Optional[Dict[str, Any]] = Field(None, description="Оценка рисков стабильности")
    security_evaluation: Optional[Dict[str, Any]] = Field(None, description="Оценка рисков безопасности")
    autonomy_evaluation: Optional[Dict[str, Any]] = Field(None, description="Оценка рисков автономности")
    regulatory_evaluation: Optional[Dict[str, Any]] = Field(None, description="Оценка регуляторных рисков")
    social_evaluation: Optional[Dict[str, Any]] = Field(None, description="Оценка социальных рисков")
    
    critic_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Результаты критического анализа")
    
    # Итоговый результат
    final_assessment: Optional[Dict[str, Any]] = Field(None, description="Итоговая оценка")
    
    # Управление процессом
    current_step: str = Field("initialization", description="Текущий шаг")
    retry_count: Dict[str, int] = Field(default_factory=dict, description="Счетчики повторов")
    max_retries: int = Field(3, description="Максимум повторов")
    
    # Настройки качества  
    quality_threshold: float = Field(7.0, description="Порог качества для критика")
    require_critic_approval: bool = Field(True, description="Требовать одобрение критика")
    
    # Контроль ошибок
    error_message: Optional[str] = Field(None, description="Сообщение об ошибке")
    retry_needed: List[str] = Field(default_factory=list, description="Риски, требующие повтора")
    
    # Временные метки
    start_time: Optional[datetime] = Field(None, description="Время начала")
    processing_time: Optional[float] = Field(None, description="Время обработки в секундах")
    
    # Статистика
    average_quality: Optional[float] = Field(None, description="Средняя оценка качества")
    
    # Сохраненные данные
    saved_assessment_id: Optional[str] = Field(None, description="ID сохраненной оценки")
    profile_id: Optional[str] = Field(None, description="ID профиля агента")
    

    # Reducer для handling concurrent updates
    @staticmethod
    def assessment_id_reducer(left: Optional[str], right: Optional[str]) -> Optional[str]:
        """Reducer для assessment_id - берем существующее значение"""
        return left if left is not None else right

    @staticmethod
    def evaluation_results_reducer(left: Dict, right: Dict) -> Dict:
        """Reducer для evaluation_results - объединяем результаты"""
        result = left.copy() if left else {}
        if right:
            result.update(right)
        return result

    class Config:
        """Конфигурация модели"""
        extra = "allow"
        use_enum_values = True
        arbitrary_types_allowed = True
        
    def __getitem__(self, key: str):
        """Поддержка доступа как к словарю"""
        return getattr(self, key, None)
    
    def __setitem__(self, key: str, value):
        """Поддержка записи как в словарь"""
        setattr(self, key, value)
    
    def get(self, key: str, default=None):
        """Метод get() как у словаря"""
        return getattr(self, key, default)
    
    def update(self, updates: Dict[str, Any]):
        """Обновление значений как у словаря"""
        for key, value in updates.items():
            setattr(self, key, value)
    
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Преобразование в словарь с поддержкой всех полей"""
        result = super().dict(**kwargs)
        
        # Добавляем все дополнительные атрибуты
        for key, value in self.__dict__.items():
            if key not in result and not key.startswith('_'):
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
        
        return result
    
    def get_evaluation_results(self) -> Dict[str, Any]:
        """ИСПРАВЛЕННАЯ версия - Собирает все результаты оценки в единый словарь"""
        
        # Собираем результаты из отдельных полей состояния
        evaluation_results = {}
        
        # Маппинг полей состояния к типам рисков
        field_mapping = {
            "ethical_evaluation": "ethical",
            "stability_evaluation": "stability", 
            "security_evaluation": "security",
            "autonomy_evaluation": "autonomy",
            "regulatory_evaluation": "regulatory",
            "social_evaluation": "social"
        }
        
        # Извлекаем результаты из состояния
        for field_name, risk_type in field_mapping.items():
            result = getattr(self, field_name, None)
            
            if result is not None:
                # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Унифицируем формат результата
                if isinstance(result, dict):
                    # Уже в правильном формате
                    evaluation_results[risk_type] = result
                elif hasattr(result, 'dict'):
                    # Pydantic модель - конвертируем в dict
                    evaluation_results[risk_type] = result.dict()
                elif hasattr(result, '__dict__'):
                    # Обычный объект - извлекаем атрибуты
                    evaluation_results[risk_type] = {
                        "status": getattr(result, 'status', 'unknown'),
                        "result_data": getattr(result, 'result_data', None),
                        "agent_name": getattr(result, 'agent_name', 'unknown'),
                        "error_message": getattr(result, 'error_message', None)
                    }
                else:
                    # Неизвестный формат - создаем минимальную структуру
                    evaluation_results[risk_type] = {
                        "status": "unknown",
                        "result_data": None,
                        "agent_name": "unknown",
                        "error_message": f"Неизвестный формат результата: {type(result)}"
                    }
        
        return evaluation_results
    
    def set_evaluation_result(self, risk_type: str, result: Any):
        """ИСПРАВЛЕННАЯ версия - Устанавливает результат оценки для конкретного типа риска"""
        
        field_mapping = {
            "ethical": "ethical_evaluation",
            "stability": "stability_evaluation", 
            "security": "security_evaluation",
            "autonomy": "autonomy_evaluation",
            "regulatory": "regulatory_evaluation",
            "social": "social_evaluation"
        }
        
        field_name = field_mapping.get(risk_type)
        if field_name:
            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Нормализуем формат перед сохранением
            normalized_result = self._normalize_evaluation_result(result)
            setattr(self, field_name, normalized_result)

    def _normalize_evaluation_result(self, result: Any) -> Dict[str, Any]:
        """Нормализует результат оценки к единому формату"""
        
        if result is None:
            return {
                "status": "failed",
                "result_data": None,
                "agent_name": "unknown",
                "error_message": "Результат оценки отсутствует"
            }
        
        # Если это уже dict в правильном формате
        if isinstance(result, dict):
            required_fields = ["status", "result_data", "agent_name"]
            if all(field in result for field in required_fields):
                return result
            
            # Если это dict но в неправильном формате, пытаемся исправить
            return {
                "status": result.get("status", "unknown"),
                "result_data": result.get("result_data", result),  # Fallback на весь result
                "agent_name": result.get("agent_name", "unknown"),
                "error_message": result.get("error_message")
            }
        
        # Если это объект AgentTaskResult или похожий
        elif hasattr(result, 'status') and hasattr(result, 'result_data'):
            return {
                "status": str(getattr(result, 'status', 'unknown')),
                "result_data": getattr(result, 'result_data', None),
                "agent_name": getattr(result, 'agent_name', 'unknown'),
                "error_message": getattr(result, 'error_message', None)
            }
        
        # Если это Pydantic модель
        elif hasattr(result, 'dict'):
            result_dict = result.dict()
            return {
                "status": result_dict.get("status", "unknown"),
                "result_data": result_dict.get("result_data", result_dict),
                "agent_name": result_dict.get("agent_name", "unknown"),
                "error_message": result_dict.get("error_message")
            }
        
        # Последний fallback - создаем минимальную структуру
        else:
            return {
                "status": "unknown",
                "result_data": str(result) if result else None,
                "agent_name": "unknown",
                "error_message": f"Неподдерживаемый тип результата: {type(result)}"
            }
    def get_successful_evaluations(self) -> Dict[str, Any]:
        """ИСПРАВЛЕННЫЙ метод - правильно обрабатывает enum статусы"""
        
        all_results = self.get_evaluation_results()
        successful_results = {}
        
        for risk_type, result in all_results.items():
            if not result:
                continue
                
            # ИСПРАВЛЕНО: Правильная проверка статуса (enum vs строка)
            status = None
            if isinstance(result, dict):
                status = result.get("status")
            elif hasattr(result, 'status'):
                status = result.status
                
            # Проверяем статус как ENUM или строку
            is_completed = False
            if hasattr(status, 'value'):
                # Это enum - проверяем value
                is_completed = status.value == "completed"
            elif str(status) == "ProcessingStatus.COMPLETED":
                # Это enum в строковом представлении
                is_completed = True  
            elif status == "completed":
                # Это строка
                is_completed = True
            
            if not is_completed:
                continue
            
            # Проверяем result_data
            result_data = None
            if isinstance(result, dict):
                result_data = result.get("result_data")
            elif hasattr(result, 'result_data'):
                result_data = result.result_data
                
            if result_data is None:
                continue
            
            successful_results[risk_type] = result
    
        return successful_results

    def get_failed_evaluations(self) -> Dict[str, Any]:
        """Возвращает только неудачные оценки"""
        
        all_results = self.get_evaluation_results()
        failed_results = {}
        
        for risk_type, result in all_results.items():
            if (not result or 
                result.get("status") in ["failed", "error"] or 
                result.get("result_data") is None):
                failed_results[risk_type] = result
        
        return failed_results

    def has_evaluation_results(self) -> bool:
        """Проверяет, есть ли хотя бы один результат оценки"""
        
        successful = self.get_successful_evaluations()
        return len(successful) > 0

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по результатам оценки"""
        
        all_results = self.get_evaluation_results()
        successful = self.get_successful_evaluations()
        failed = self.get_failed_evaluations()
        
        return {
            "total_evaluations": len(all_results),
            "successful_evaluations": len(successful),
            "failed_evaluations": len(failed),
            "success_rate": len(successful) / len(all_results) if all_results else 0,
            "successful_types": list(successful.keys()),
            "failed_types": list(failed.keys())
        }
# ===============================
# Вспомогательные функции
# ===============================

def create_risk_evaluation(
    risk_type: RiskType,
    probability: int,
    impact: int,
    probability_reasoning: str,
    impact_reasoning: str,
    evaluator_agent: str,
    recommendations: List[str] = None,
    confidence: float = 0.8
) -> RiskEvaluation:
    """Создание оценки риска с автоматическим расчетом"""
    
    return RiskEvaluation(
        risk_type=risk_type,
        evaluator_agent=evaluator_agent,
        probability_score=probability,
        impact_score=impact,
        probability_reasoning=probability_reasoning,
        impact_reasoning=impact_reasoning,
        recommendations=recommendations or [],
        confidence_level=confidence
    )


def calculate_overall_risk_score(risk_evaluations: Dict[str, RiskEvaluation]) -> tuple[int, RiskLevel]:
    """Расчет общего балла и уровня риска"""
    
    if not risk_evaluations:
        return 1, RiskLevel.LOW
    
    total_scores = [eval.total_score for eval in risk_evaluations.values()]
    overall_score = sum(total_scores) // len(total_scores)
    
    if overall_score <= 6:
        return overall_score, RiskLevel.LOW
    elif overall_score <= 14:
        return overall_score, RiskLevel.MEDIUM
    else:
        return overall_score, RiskLevel.HIGH


def extract_risk_evaluations_from_results(evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, RiskEvaluation]:
    """Извлечение RiskEvaluation из результатов агентов"""
    
    risk_evaluations = {}
    
    for risk_type, result in evaluation_results.items():
        if isinstance(result, dict) and result.get("result_data"):
            risk_eval_data = result["result_data"].get("risk_evaluation")
            if risk_eval_data:
                try:
                    risk_evaluations[risk_type] = RiskEvaluation(**risk_eval_data)
                except Exception as e:
                    print(f"Ошибка создания RiskEvaluation для {risk_type}: {e}")
    
    return risk_evaluations


def get_highest_risk_areas(risk_evaluations: Dict[str, RiskEvaluation], top_n: int = 3) -> List[str]:
    """Получение областей наивысшего риска"""
    
    if not risk_evaluations:
        return []
    
    # Сортируем по общему баллу
    sorted_risks = sorted(
        risk_evaluations.items(),
        key=lambda x: x[1].total_score,
        reverse=True
    )
    
    return [risk_type for risk_type, _ in sorted_risks[:top_n]]


def validate_workflow_state(state: Dict[str, Any]) -> WorkflowState:
    """Валидация и создание WorkflowState из словаря"""
    
    try:
        return WorkflowState(**state)
    except Exception as e:
        print(f"Ошибка валидации WorkflowState: {e}")
        # Возвращаем минимальное валидное состояние
        return WorkflowState(
            source_files=state.get("source_files", []),
            current_step=state.get("current_step", "error"),
            error_message=f"Ошибка валидации: {str(e)}"
        )


# ===============================
# Экспорт
# ===============================

__all__ = [
    # Перечисления
    "RiskType", "RiskLevel", "AgentType", "AutonomyLevel", 
    "DataSensitivity", "ProcessingStatus",
    
    # Основные модели
    "AgentProfile", "RiskEvaluation", "CriticEvaluation",
    "AgentRiskAssessment", "AgentTaskResult", "WorkflowState",
    
    # Вспомогательные функции
    "create_risk_evaluation", "calculate_overall_risk_score",
    "extract_risk_evaluations_from_results", "get_highest_risk_areas",
    "validate_workflow_state"
]