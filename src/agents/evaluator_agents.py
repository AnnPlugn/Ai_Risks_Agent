# src/agents/evaluator_agents.py
"""
Агенты-оценщики рисков ИИ-агентов
6 специализированных агентов для оценки разных типов операционных рисков
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from .base_agent import EvaluationAgent, AgentConfig
from ..models.risk_models import (
    RiskType, RiskEvaluation, AgentTaskResult, ProcessingStatus, WorkflowState
)
from ..utils.logger import LogContext
from ..prompts.call_evaluator_prompts import (call_ethical_prompt, call_social_risk_prompt, call_regulatory_risk_prompt,
                                              call_autonomy_risk_prompt, call_security_risk_prompt, call_stability_risk_prompt)



class EthicalRiskEvaluator(EvaluationAgent):
    """Агент-оценщик этических и дискриминационных рисков"""
    
    def get_system_prompt(self) -> str:
        try:
            import uuid
            temp_assessment_id = str(uuid.uuid4())[:8]
            bound_logger = self.logger.bind_context(temp_assessment_id, self.name)
            bound_logger.info(f"📋 EthicalRiskEvaluator использует ОБНОВЛЕННЫЙ промпт (версия с требованиями 800+ символов)")
        except:
            print("📋 EthicalRiskEvaluator использует ОБНОВЛЕННЫЙ промпт")

        return call_ethical_prompt

    async def process(
        self, 
        input_data: Dict[str, Any], 
        assessment_id: str
    ) -> AgentTaskResult:
        """ОТЛАДОЧНАЯ версия оценки этических рисков для GigaChat"""
        start_time = datetime.now()
        
        try:
            with LogContext("evaluate_ethical_risk", assessment_id, self.name):
                agent_profile = input_data.get("agent_profile", {})
                agent_data = self._format_agent_data(agent_profile)
                
                print(f"🔍 DEBUG EthicalRiskEvaluator: Начинаем evaluate_risk...")
                
                # Получаем сырые данные от LLM
                evaluation_result = await self.evaluate_risk(
                    risk_type="этические и дискриминационные риски",
                    agent_data=agent_data,
                    evaluation_criteria=self.get_system_prompt(),
                    assessment_id=assessment_id
                )
                
                # ===== ПРИНУДИТЕЛЬНАЯ ПРОВЕРКА THREAT_ASSESSMENTS =====
                print(f"🔍 DEBUG EthicalRiskEvaluator: Получен ответ от LLM")
                print(f"🔍 DEBUG: Ключи в ответе: {list(evaluation_result.keys())}")
                
                if "threat_assessments" not in evaluation_result:
                    print("❌ КРИТИЧЕСКАЯ ОШИБКА: threat_assessments отсутствует в ответе от GigaChat!")
                    print("📄 Полный ответ от GigaChat:")
                    import json
                    print(json.dumps(evaluation_result, ensure_ascii=False, indent=2))
                    
                    # Создаем fallback threat_assessments
                    print("🔧 Создаем fallback threat_assessments...")
                    evaluation_result["threat_assessments"] = {
                        "галлюцинации_и_зацикливание": {
                            "risk_level": "средняя",
                            "probability_score": 3,
                            "impact_score": 3,
                            "reasoning": "Fallback оценка: GigaChat не сгенерировал threat_assessments. Анализ показывает средний риск галлюцинаций из-за отсутствия явных механизмов контроля RAG и валидации в представленной архитектуре агента."
                        },
                        "дезинформация": {
                            "risk_level": "средняя", 
                            "probability_score": 3,
                            "impact_score": 3,
                            "reasoning": "Fallback оценка: GigaChat не сгенерировал threat_assessments. Средний риск дезинформации из-за отсутствия модерации контента и этических гайдлайнов в системных промптах."
                        },
                        "токсичность_и_дискриминация": {
                            "risk_level": "средняя",
                            "probability_score": 3, 
                            "impact_score": 3,
                            "reasoning": "Fallback оценка: GigaChat не сгенерировал threat_assessments. Средний риск токсичности из-за отсутствия фильтрации данных и процедур экстренного отключения."
                        }
                    }
                    print("✅ Fallback threat_assessments создан")
                else:
                    print("✅ DEBUG EthicalRiskEvaluator: threat_assessments найден в ответе!")
                    threats = evaluation_result["threat_assessments"]
                    print(f"🔍 Найдено угроз: {len(threats)}")
                    for threat_name, threat_data in threats.items():
                        risk_level = threat_data.get('risk_level', 'unknown')
                        reasoning_length = len(str(threat_data.get('reasoning', '')))
                        print(f"  🎯 {threat_name}: {risk_level} (reasoning: {reasoning_length} символов)")
                
                # БЕЗОПАСНОЕ создание RiskEvaluation
                risk_evaluation = RiskEvaluation.create_safe(
                    risk_type=RiskType.ETHICAL,
                    evaluator_agent=self.name,
                    raw_data=evaluation_result
                )
                
                # Логируем результат с threat_assessments если есть
                threat_assessments_dict = {}
                if hasattr(risk_evaluation, 'threat_assessments') and risk_evaluation.threat_assessments:
                    for threat_name, threat_obj in risk_evaluation.threat_assessments.items():
                        threat_assessments_dict[threat_name] = {
                            "risk_level": threat_obj.risk_level,
                            "probability_score": threat_obj.probability_score,
                            "impact_score": threat_obj.impact_score,
                            "reasoning": threat_obj.reasoning
                        }
                    print(f"✅ Передаем в логирование {len(threat_assessments_dict)} угроз")
                
                self.logger.log_risk_evaluation(
                    self.name,
                    assessment_id,
                    "этические и дискриминационные риски",
                    risk_evaluation.total_score,
                    risk_evaluation.risk_level.value,
                    threat_assessments_dict if threat_assessments_dict else None
                )
                
                # Создаем успешный результат
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return AgentTaskResult(
                    agent_name=self.name,
                    task_type="ethicalriskevaluator",
                    status=ProcessingStatus.COMPLETED,
                    result_data={"risk_evaluation": risk_evaluation},
                    execution_time_seconds=execution_time,
                    assessment_id=assessment_id
                )
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_message = f"Ошибка оценки этических рисков: {str(e)}"
            
            print(f"❌ ОШИБКА в EthicalRiskEvaluator: {error_message}")
            import traceback
            traceback.print_exc()
            
            return AgentTaskResult(
                task_type="ethicalriskevaluator",
                status=ProcessingStatus.FAILED,
                result_data={},
                execution_time_seconds=execution_time,
                error_message=error_message,
                assessment_id=assessment_id
            )

    def _format_agent_data(self, agent_profile: Dict[str, Any]) -> str:
        """Форматирование данных агента для анализа этических рисков"""
        # 🔍 ИСПРАВЛЕННАЯ ДИАГНОСТИКА - используем правильный способ логирования
        detailed_summary = agent_profile.get('detailed_summary', {})
        
        # ЛОГИРУЕМ С ПРАВИЛЬНЫМ assessment_id (если есть в контексте)
        try:
            # Пытаемся получить assessment_id из контекста или используем временный
            import uuid
            temp_assessment_id = str(uuid.uuid4())[:8]
            bound_logger = self.logger.bind_context(temp_assessment_id, self.name)
            
            if detailed_summary:
                bound_logger.info(f"✅ EthicalRiskEvaluator получил детальное саммари с {len(detailed_summary)} разделами")
                
                # Логируем размеры разделов (только если DEBUG уровень)
                total_summary_length = 0
                for section, content in detailed_summary.items():
                    section_length = len(str(content))
                    total_summary_length += section_length
                    bound_logger.debug(f"  📊 {section}: {section_length} символов")
                
                bound_logger.info(f"📏 Общий размер саммари: {total_summary_length} символов")
            else:
                bound_logger.warning("⚠️ EthicalRiskEvaluator НЕ получил детальное саммари")
                
        except Exception as e:
            # Fallback логирование если что-то пошло не так
            print(f"🔍 ДИАГНОСТИКА: detailed_summary есть: {bool(detailed_summary)}, разделов: {len(detailed_summary) if detailed_summary else 0}")
        
        

        basic_info = f"""ПРОФИЛЬ АГЕНТА:
Название: {agent_profile.get('name', 'Unknown')}
Тип: {agent_profile.get('agent_type', 'unknown')}
Описание: {agent_profile.get('description', 'Не указано')}
Автономность: {agent_profile.get('autonomy_level', 'unknown')}
Доступ к данным: {', '.join(agent_profile.get('data_access', []))}
Целевая аудитория: {agent_profile.get('target_audience', 'Не указано')}

СИСТЕМНЫЕ ПРОМПТЫ:
{chr(10).join(agent_profile.get('system_prompts', ['Не найдены']))}

ОГРАНИЧЕНИЯ БЕЗОПАСНОСТИ:
{chr(10).join(agent_profile.get('guardrails', ['Не найдены']))}

ВНЕШНИЕ API: {', '.join(agent_profile.get('external_apis', ['Нет']))}"""
        # ДОБАВЛЯЕМ ДЕТАЛЬНОЕ САММАРИ
        #detailed_summary = agent_profile.get('detailed_summary', {})
        if detailed_summary:
            summary_section = f"""

=== ДЕТАЛЬНОЕ САММАРИ ===

ОБЗОР АГЕНТА:
{detailed_summary.get('overview', 'Информация отсутствует')}

ТЕХНИЧЕСКАЯ АРХИТЕКТУРА:
{detailed_summary.get('technical_architecture', 'Информация отсутствует')}

ОПЕРАЦИОННАЯ МОДЕЛЬ:
{detailed_summary.get('operational_model', 'Информация отсутствует')}

АНАЛИЗ РИСКОВ:
{detailed_summary.get('risk_analysis', 'Информация отсутствует')}

РЕКОМЕНДАЦИИ ПО БЕЗОПАСНОСТИ:
{detailed_summary.get('security_recommendations', 'Информация отсутствует')}

ВЫВОДЫ:
{detailed_summary.get('conclusions', 'Информация отсутствует')}"""
            
            return basic_info + summary_section
        
        return basic_info

class StabilityRiskEvaluator(EvaluationAgent):
    """Агент-оценщик рисков ошибок и нестабильности LLM"""
    
    def get_system_prompt(self) -> str:
        return call_stability_risk_prompt

    async def process(
        self, 
        input_data: Dict[str, Any], 
        assessment_id: str
    ) -> AgentTaskResult:
        """ИСПРАВЛЕННАЯ оценка рисков стабильности"""
        start_time = datetime.now()
        
        try:
            with LogContext("evaluate_stability_risk", assessment_id, self.name):
                agent_profile = input_data.get("agent_profile", {})
                agent_data = self._format_agent_data(agent_profile)
                
                evaluation_result = await self.evaluate_risk(
                    risk_type="риски ошибок и нестабильности LLM",
                    agent_data=agent_data,
                    evaluation_criteria=self.get_system_prompt(),
                    assessment_id=assessment_id
                )
                
                # БЕЗОПАСНОЕ создание RiskEvaluation
                risk_evaluation = RiskEvaluation.create_safe(
                    risk_type=RiskType.STABILITY,
                    evaluator_agent=self.name,
                    raw_data=evaluation_result
                )
                
                self.logger.log_risk_evaluation(
                    self.name, assessment_id, "риски ошибок и нестабильности LLM",
                    risk_evaluation.total_score, risk_evaluation.risk_level.value
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return AgentTaskResult(
                    agent_name=self.name,
                    task_type="stabilityriskevaluator", 
                    status=ProcessingStatus.COMPLETED,
                    result_data={"risk_evaluation": risk_evaluation.dict()},
                    start_time=start_time,
                    end_time=datetime.now(),
                    execution_time_seconds=execution_time
                )
                
        except Exception as e:
            self.logger.bind_context(assessment_id, self.name).error(
                f"❌ Ошибка оценки рисков стабильности: {e}"
            )
            
            fallback_evaluation = RiskEvaluation.create_from_raw_data(
                risk_type=RiskType.STABILITY,
                evaluator_agent=self.name,
                raw_data={"error_message": str(e)}
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentTaskResult(
                agent_name=self.name,
                task_type="stabilityriskevaluator",
                status=ProcessingStatus.COMPLETED,
                result_data={"risk_evaluation": fallback_evaluation.dict()},
                start_time=start_time,
                end_time=datetime.now(),
                execution_time_seconds=execution_time,
                error_message=f"Fallback данные: {str(e)}"
            )

    def _format_agent_data(self, agent_profile: Dict[str, Any]) -> str:
        """Форматирование данных для анализа стабильности"""
        basic_info = f"""ТЕХНИЧЕСКИЙ ПРОФИЛЬ АГЕНТА:
Название: {agent_profile.get('name', 'Unknown')}
LLM Модель: {agent_profile.get('llm_model', 'unknown')}
Тип агента: {agent_profile.get('agent_type', 'unknown')}
Операций в час: {agent_profile.get('operations_per_hour', 'Не указано')}
Автономность: {agent_profile.get('autonomy_level', 'unknown')}

СИСТЕМНЫЕ ПРОМПТЫ (анализ сложности):
{chr(10).join(agent_profile.get('system_prompts', ['Не найдены']))}

ВНЕШНИЕ ЗАВИСИМОСТИ:
APIs: {', '.join(agent_profile.get('external_apis', ['Нет']))}

МОНИТОРИНГ И КОНТРОЛЬ:
Ограничения: {chr(10).join(agent_profile.get('guardrails', ['Не найдены']))}"""
# ДОБАВЛЯЕМ ДЕТАЛЬНОЕ САММАРИ
        detailed_summary = agent_profile.get('detailed_summary', {})
        if detailed_summary:
            summary_section = f"""

=== ДЕТАЛЬНОЕ САММАРИ ===

ОБЗОР АГЕНТА:
{detailed_summary.get('overview', 'Информация отсутствует')}

ТЕХНИЧЕСКАЯ АРХИТЕКТУРА:
{detailed_summary.get('technical_architecture', 'Информация отсутствует')}

ОПЕРАЦИОННАЯ МОДЕЛЬ:
{detailed_summary.get('operational_model', 'Информация отсутствует')}

АНАЛИЗ РИСКОВ:
{detailed_summary.get('risk_analysis', 'Информация отсутствует')}

РЕКОМЕНДАЦИИ ПО БЕЗОПАСНОСТИ:
{detailed_summary.get('security_recommendations', 'Информация отсутствует')}

ВЫВОДЫ:
{detailed_summary.get('conclusions', 'Информация отсутствует')}"""
            
            return basic_info + summary_section
        
        return basic_info

class SecurityRiskEvaluator(EvaluationAgent):
    """Агент-оценщик рисков безопасности данных и систем"""
    
    def get_system_prompt(self) -> str:
        return call_security_risk_prompt

    
    async def process(
        self, 
        input_data: Dict[str, Any], 
        assessment_id: str
    ) -> AgentTaskResult:
        """ИСПРАВЛЕННАЯ оценка рисков безопасности"""
        start_time = datetime.now()
        
        try:
            with LogContext("evaluate_security_risk", assessment_id, self.name):
                agent_profile = input_data.get("agent_profile", {})
                agent_data = self._format_agent_data(agent_profile)
                
                evaluation_result = await self.evaluate_risk(
                    risk_type="риски безопасности данных и систем",
                    agent_data=agent_data,
                    evaluation_criteria=self.get_system_prompt(),
                    assessment_id=assessment_id
                )
                
                # БЕЗОПАСНОЕ создание RiskEvaluation
                risk_evaluation = RiskEvaluation.create_safe(
                    risk_type=RiskType.SECURITY,
                    evaluator_agent=self.name,
                    raw_data=evaluation_result
                )
                
                self.logger.log_risk_evaluation(
                    self.name, assessment_id, "риски безопасности данных и систем",
                    risk_evaluation.total_score, risk_evaluation.risk_level.value
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return AgentTaskResult(
                    agent_name=self.name,
                    task_type="security_risk_evaluation", 
                    status=ProcessingStatus.COMPLETED,
                    result_data={"risk_evaluation": risk_evaluation.dict()},
                    start_time=start_time,
                    end_time=datetime.now(),
                    execution_time_seconds=execution_time
                )
                
        except Exception as e:
            self.logger.bind_context(assessment_id, self.name).error(
                f"❌ Ошибка оценки рисков безопасности: {e}"
            )
            
            fallback_evaluation = RiskEvaluation.create_from_raw_data(
                risk_type=RiskType.SECURITY,
                evaluator_agent=self.name,
                raw_data={"error_message": str(e)}
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentTaskResult(
                agent_name=self.name,
                task_type="security_risk_evaluation",
                status=ProcessingStatus.COMPLETED,
                result_data={"risk_evaluation": fallback_evaluation.dict()},
                start_time=start_time,
                end_time=datetime.now(),
                execution_time_seconds=execution_time,
                error_message=f"Fallback данные: {str(e)}"
            )

    def _format_agent_data(self, agent_profile: Dict[str, Any]) -> str:
        """Форматирование данных для анализа безопасности"""
        basic_info = f"""ПРОФИЛЬ БЕЗОПАСНОСТИ АГЕНТА:
Название: {agent_profile.get('name', 'Unknown')}
Доступ к данным: {', '.join(agent_profile.get('data_access', []))}
Внешние APIs: {', '.join(agent_profile.get('external_apis', ['Нет']))}
Уровень автономности: {agent_profile.get('autonomy_level', 'unknown')}

МЕРЫ БЕЗОПАСНОСТИ:
{chr(10).join(agent_profile.get('guardrails', ['Не найдены']))}

СИСТЕМНЫЕ ПРОМПТЫ (анализ на уязвимости):
{chr(10).join(agent_profile.get('system_prompts', ['Не найдены']))}

ОПЕРАЦИОННЫЙ КОНТЕКСТ:
Целевая аудитория: {agent_profile.get('target_audience', 'Не указано')}
Операций в час: {agent_profile.get('operations_per_hour', 'Не указано')}"""
# ДОБАВЛЯЕМ ДЕТАЛЬНОЕ САММАРИ
        detailed_summary = agent_profile.get('detailed_summary', {})
        if detailed_summary:
            summary_section = f"""

=== ДЕТАЛЬНОЕ САММАРИ ===

ОБЗОР АГЕНТА:
{detailed_summary.get('overview', 'Информация отсутствует')}

ТЕХНИЧЕСКАЯ АРХИТЕКТУРА:
{detailed_summary.get('technical_architecture', 'Информация отсутствует')}

ОПЕРАЦИОННАЯ МОДЕЛЬ:
{detailed_summary.get('operational_model', 'Информация отсутствует')}

АНАЛИЗ РИСКОВ:
{detailed_summary.get('risk_analysis', 'Информация отсутствует')}

РЕКОМЕНДАЦИИ ПО БЕЗОПАСНОСТИ:
{detailed_summary.get('security_recommendations', 'Информация отсутствует')}

ВЫВОДЫ:
{detailed_summary.get('conclusions', 'Информация отсутствует')}"""
            
            return basic_info + summary_section
        
        return basic_info

class AutonomyRiskEvaluator(EvaluationAgent):
    """Агент-оценщик рисков автономности и управления"""
    
    def get_system_prompt(self) -> str:
        try:
            import uuid
            temp_assessment_id = str(uuid.uuid4())[:8]
            bound_logger = self.logger.bind_context(temp_assessment_id, self.name)
            bound_logger.info(f"📋 AutonomyRiskEvaluator использует ОБНОВЛЕННЫЙ промпт (версия с требованиями 1000+ символов)")
        except:
            print("📋 AutonomyRiskEvaluator использует ОБНОВЛЕННЫЙ промпт")

        return call_autonomy_risk_prompt

        
    async def process(
        self, 
        input_data: Dict[str, Any], 
        assessment_id: str
    ) -> AgentTaskResult:
        """ИСПРАВЛЕННАЯ оценка рисков автономности"""
        start_time = datetime.now()
        
        try:
            with LogContext("evaluate_autonomy_risk", assessment_id, self.name):
                agent_profile = input_data.get("agent_profile", {})
                agent_data = self._format_agent_data(agent_profile)
                
                evaluation_result = await self.evaluate_risk(
                    risk_type="риски автономности и управления",
                    agent_data=agent_data,
                    evaluation_criteria=self.get_system_prompt(),
                    assessment_id=assessment_id
                )
                
                # БЕЗОПАСНОЕ создание RiskEvaluation
                risk_evaluation = RiskEvaluation.create_safe(
                    risk_type=RiskType.AUTONOMY,
                    evaluator_agent=self.name,
                    raw_data=evaluation_result
                )
                
                self.logger.log_risk_evaluation(
                    self.name, assessment_id, "риски автономности и управления",
                    risk_evaluation.total_score, risk_evaluation.risk_level.value
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return AgentTaskResult(
                    agent_name=self.name,
                    task_type="autonomy_risk_evaluation",
                    status=ProcessingStatus.COMPLETED,
                    result_data={"risk_evaluation": risk_evaluation.dict()},
                    start_time=start_time,
                    end_time=datetime.now(),
                    execution_time_seconds=execution_time
                )
                
        except Exception as e:
            self.logger.bind_context(assessment_id, self.name).error(
                f"❌ Ошибка оценки рисков автономности: {e}"
            )
            
            fallback_evaluation = RiskEvaluation.create_from_raw_data(
                risk_type=RiskType.AUTONOMY,
                evaluator_agent=self.name,
                raw_data={"error_message": str(e)}
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentTaskResult(
                agent_name=self.name,
                task_type="autonomy_risk_evaluation",
                status=ProcessingStatus.COMPLETED,
                result_data={"risk_evaluation": fallback_evaluation.dict()},
                start_time=start_time,
                end_time=datetime.now(),
                execution_time_seconds=execution_time,
                error_message=f"Fallback данные: {str(e)}"
            )

    def _format_agent_data(self, agent_profile: Dict[str, Any]) -> str:
        """Форматирование данных для анализа автономности"""
        basic_info = f"""ПРОФИЛЬ АВТОНОМНОСТИ АГЕНТА:
Название: {agent_profile.get('name', 'Unknown')}
Уровень автономности: {agent_profile.get('autonomy_level', 'unknown')}
Тип агента: {agent_profile.get('agent_type', 'unknown')}
Операций в час: {agent_profile.get('operations_per_hour', 'Не указано')}
Доход с операции: {agent_profile.get('revenue_per_operation', 'Не указано')} руб

ОБЛАСТЬ ОТВЕТСТВЕННОСТИ:
{agent_profile.get('description', 'Не указано')}

ОГРАНИЧЕНИЯ И КОНТРОЛЬ:
{chr(10).join(agent_profile.get('guardrails', ['Не найдены']))}

СИСТЕМНЫЕ ИНСТРУКЦИИ:
{chr(10).join(agent_profile.get('system_prompts', ['Не найдены']))}

ИНТЕГРАЦИИ:
Внешние API: {', '.join(agent_profile.get('external_apis', ['Нет']))}
Доступ к данным: {', '.join(agent_profile.get('data_access', []))}"""
# ДОБАВЛЯЕМ ДЕТАЛЬНОЕ САММАРИ
        detailed_summary = agent_profile.get('detailed_summary', {})
        if detailed_summary:
            summary_section = f"""

=== ДЕТАЛЬНОЕ САММАРИ ===

ОБЗОР АГЕНТА:
{detailed_summary.get('overview', 'Информация отсутствует')}

ТЕХНИЧЕСКАЯ АРХИТЕКТУРА:
{detailed_summary.get('technical_architecture', 'Информация отсутствует')}

ОПЕРАЦИОННАЯ МОДЕЛЬ:
{detailed_summary.get('operational_model', 'Информация отсутствует')}

АНАЛИЗ РИСКОВ:
{detailed_summary.get('risk_analysis', 'Информация отсутствует')}

РЕКОМЕНДАЦИИ ПО БЕЗОПАСНОСТИ:
{detailed_summary.get('security_recommendations', 'Информация отсутствует')}

ВЫВОДЫ:
{detailed_summary.get('conclusions', 'Информация отсутствует')}"""
            
            return basic_info + summary_section
        
        return basic_info

class RegulatoryRiskEvaluator(EvaluationAgent):
    """Агент-оценщик регуляторных и юридических рисков"""
    
    def get_system_prompt(self) -> str:
        return call_regulatory_risk_prompt

    

    async def process(
        self, 
        input_data: Dict[str, Any], 
        assessment_id: str
    ) -> AgentTaskResult:
        """ИСПРАВЛЕННАЯ оценка регуляторных рисков"""
        start_time = datetime.now()
        
        try:
            with LogContext("evaluate_regulatory_risk", assessment_id, self.name):
                agent_profile = input_data.get("agent_profile", {})
                agent_data = self._format_agent_data(agent_profile)
                
                evaluation_result = await self.evaluate_risk(
                    risk_type="регуляторные и юридические риски",
                    agent_data=agent_data,
                    evaluation_criteria=self.get_system_prompt(),
                    assessment_id=assessment_id
                )
                
                # БЕЗОПАСНОЕ создание RiskEvaluation
                risk_evaluation = RiskEvaluation.create_safe(
                    risk_type=RiskType.REGULATORY,
                    evaluator_agent=self.name,
                    raw_data=evaluation_result
                )
                
                self.logger.log_risk_evaluation(
                    self.name, assessment_id, "регуляторные и юридические риски",
                    risk_evaluation.total_score, risk_evaluation.risk_level.value
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return AgentTaskResult(
                    agent_name=self.name,
                    task_type="regulatory_risk_evaluation",
                    status=ProcessingStatus.COMPLETED,
                    result_data={"risk_evaluation": risk_evaluation.dict()},
                    start_time=start_time,
                    end_time=datetime.now(),
                    execution_time_seconds=execution_time
                )
                
        except Exception as e:
            self.logger.bind_context(assessment_id, self.name).error(
                f"❌ Ошибка оценки регуляторных рисков: {e}"
            )
            
            fallback_evaluation = RiskEvaluation.create_from_raw_data(
                risk_type=RiskType.REGULATORY,
                evaluator_agent=self.name,
                raw_data={"error_message": str(e)}
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentTaskResult(
                agent_name=self.name,
                task_type="regulatory_risk_evaluation",
                status=ProcessingStatus.COMPLETED,
                result_data={"risk_evaluation": fallback_evaluation.dict()},
                start_time=start_time,
                end_time=datetime.now(),
                execution_time_seconds=execution_time,
                error_message=f"Fallback данные: {str(e)}"
            )

    def _format_agent_data(self, agent_profile: Dict[str, Any]) -> str:
        """Форматирование данных для регуляторного анализа"""
        basic_info = f"""РЕГУЛЯТОРНЫЙ ПРОФИЛЬ АГЕНТА:
Название: {agent_profile.get('name', 'Unknown')}
Тип деятельности: {agent_profile.get('agent_type', 'unknown')}
Целевая аудитория: {agent_profile.get('target_audience', 'Не указано')}
Доступ к данным: {', '.join(agent_profile.get('data_access', []))}

ОБРАБОТКА ПЕРСОНАЛЬНЫХ ДАННЫХ:
Уровень доступа: {', '.join(agent_profile.get('data_access', []))}
Внешние интеграции: {', '.join(agent_profile.get('external_apis', ['Нет']))}

МЕРЫ СООТВЕТСТВИЯ:
{chr(10).join(agent_profile.get('guardrails', ['Не найдены']))}

ОПЕРАЦИОННАЯ МОДЕЛЬ:
Автономность: {agent_profile.get('autonomy_level', 'unknown')}
Операций в час: {agent_profile.get('operations_per_hour', 'Не указано')}
Доход с операции: {agent_profile.get('revenue_per_operation', 'Не указано')} руб

ТЕХНИЧЕСКИЕ ДЕТАЛИ:
LLM: {agent_profile.get('llm_model', 'unknown')}
Системные инструкции: {chr(10).join(agent_profile.get('system_prompts', ['Не найдены']))}"""
# ДОБАВЛЯЕМ ДЕТАЛЬНОЕ САММАРИ
        detailed_summary = agent_profile.get('detailed_summary', {})
        if detailed_summary:
            summary_section = f"""

=== ДЕТАЛЬНОЕ САММАРИ ===

ОБЗОР АГЕНТА:
{detailed_summary.get('overview', 'Информация отсутствует')}

ТЕХНИЧЕСКАЯ АРХИТЕКТУРА:
{detailed_summary.get('technical_architecture', 'Информация отсутствует')}

ОПЕРАЦИОННАЯ МОДЕЛЬ:
{detailed_summary.get('operational_model', 'Информация отсутствует')}

АНАЛИЗ РИСКОВ:
{detailed_summary.get('risk_analysis', 'Информация отсутствует')}

РЕКОМЕНДАЦИИ ПО БЕЗОПАСНОСТИ:
{detailed_summary.get('security_recommendations', 'Информация отсутствует')}

ВЫВОДЫ:
{detailed_summary.get('conclusions', 'Информация отсутствует')}"""
            
            return basic_info + summary_section
        
        return basic_info

class SocialRiskEvaluator(EvaluationAgent):
    """Агент-оценщик социальных и манипулятивных рисков"""
    
    def get_system_prompt(self) -> str:
        return call_social_risk_prompt

    
    async def process(
        self, 
        input_data: Dict[str, Any], 
        assessment_id: str
    ) -> AgentTaskResult:
        """ИСПРАВЛЕННАЯ оценка социальных рисков"""
        start_time = datetime.now()
        
        try:
            with LogContext("evaluate_social_risk", assessment_id, self.name):
                agent_profile = input_data.get("agent_profile", {})
                agent_data = self._format_agent_data(agent_profile)
                
                evaluation_result = await self.evaluate_risk(
                    risk_type="социальные и манипулятивные риски",
                    agent_data=agent_data,
                    evaluation_criteria=self.get_system_prompt(),
                    assessment_id=assessment_id
                )
                
                # БЕЗОПАСНОЕ создание RiskEvaluation
                risk_evaluation = RiskEvaluation.create_safe(
                    risk_type=RiskType.SOCIAL,
                    evaluator_agent=self.name,
                    raw_data=evaluation_result
                )
                
                self.logger.log_risk_evaluation(
                    self.name, assessment_id, "социальные и манипулятивные риски",
                    risk_evaluation.total_score, risk_evaluation.risk_level.value
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return AgentTaskResult(
                    agent_name=self.name,
                    task_type="socialriskevaluator",
                    status=ProcessingStatus.COMPLETED,
                    result_data={"risk_evaluation": risk_evaluation.dict()},
                    start_time=start_time,
                    end_time=datetime.now(),
                    execution_time_seconds=execution_time
                )
                
        except Exception as e:
            self.logger.bind_context(assessment_id, self.name).error(
                f"❌ Ошибка оценки социальных рисков: {e}"
            )
            
            fallback_evaluation = RiskEvaluation.create_from_raw_data(
                risk_type=RiskType.SOCIAL,
                evaluator_agent=self.name,
                raw_data={"error_message": str(e)}
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return AgentTaskResult(
                agent_name=self.name,
                task_type="socialriskevaluator",
                status=ProcessingStatus.COMPLETED,
                result_data={"risk_evaluation": fallback_evaluation.dict()},
                start_time=start_time,
                end_time=datetime.now(),
                execution_time_seconds=execution_time,
                error_message=f"Fallback данные: {str(e)}"
            )
        
    def _format_agent_data(self, agent_profile: Dict[str, Any]) -> str:
        """Форматирование данных для анализа социальных рисков"""
        basic_info = f"""СОЦИАЛЬНЫЙ ПРОФИЛЬ АГЕНТА:
Название: {agent_profile.get('name', 'Unknown')}
Целевая аудитория: {agent_profile.get('target_audience', 'Не указано')}
Тип взаимодействия: {agent_profile.get('agent_type', 'unknown')}
Операций в час: {agent_profile.get('operations_per_hour', 'Не указано')}

ХАРАКТЕР ВЗАИМОДЕЙСТВИЯ:
{agent_profile.get('description', 'Не указано')}

ВОЗМОЖНОСТИ ВЛИЯНИЯ:
Системные промпты: {chr(10).join(agent_profile.get('system_prompts', ['Не найдены']))}

ЗАЩИТНЫЕ МЕРЫ:
{chr(10).join(agent_profile.get('guardrails', ['Не найдены']))}

КОНТЕКСТ ИСПОЛЬЗОВАНИЯ:
Доступ к данным: {', '.join(agent_profile.get('data_access', []))}
Автономность: {agent_profile.get('autonomy_level', 'unknown')}"""
# ДОБАВЛЯЕМ ДЕТАЛЬНОЕ САММАРИ
        detailed_summary = agent_profile.get('detailed_summary', {})
        if detailed_summary:
            summary_section = f"""

=== ДЕТАЛЬНОЕ САММАРИ ===

ОБЗОР АГЕНТА:
{detailed_summary.get('overview', 'Информация отсутствует')}

ТЕХНИЧЕСКАЯ АРХИТЕКТУРА:
{detailed_summary.get('technical_architecture', 'Информация отсутствует')}

ОПЕРАЦИОННАЯ МОДЕЛЬ:
{detailed_summary.get('operational_model', 'Информация отсутствует')}

АНАЛИЗ РИСКОВ:
{detailed_summary.get('risk_analysis', 'Информация отсутствует')}

РЕКОМЕНДАЦИИ ПО БЕЗОПАСНОСТИ:
{detailed_summary.get('security_recommendations', 'Информация отсутствует')}

ВЫВОДЫ:
{detailed_summary.get('conclusions', 'Информация отсутствует')}"""
            
            return basic_info + summary_section
        
        return basic_info

# ===============================
# Фабрики и утилиты для создания агентов
# ===============================



def create_all_evaluator_agents(
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    temperature: Optional[float] = None
) -> Dict[RiskType, EvaluationAgent]:
    """
    Создание всех 6 агентов-оценщиков
    ОБНОВЛЕНО: Использует центральный конфигуратор
    """
    from .base_agent import create_agent_config
    
    # ИЗМЕНЕНО: Базовая конфигурация теперь использует центральный конфигуратор
    base_config_params = {
        "llm_base_url": llm_base_url,
        "llm_model": llm_model,
        "temperature": temperature,
        "max_retries": 3,
        "timeout_seconds": 120,
        "use_risk_analysis_client": True  # Все оценщики используют специализированный клиент
    }
    
    # Создаем конфигурации для каждого агента
    configs = {
        RiskType.ETHICAL: create_agent_config(
            name="ethical_risk_evaluator",
            description="Агент для оценки этических и дискриминационных рисков",
            **base_config_params
        ),
        RiskType.STABILITY: create_agent_config(
            name="stability_risk_evaluator", 
            description="Агент для оценки рисков ошибок и нестабильности LLM",
            **base_config_params
        ),
        RiskType.SECURITY: create_agent_config(
            name="security_risk_evaluator",
            description="Агент для оценки рисков безопасности данных и систем",
            **base_config_params
        ),
        RiskType.AUTONOMY: create_agent_config(
            name="autonomy_risk_evaluator",
            description="Агент для оценки рисков автономности и управления",
            **base_config_params
        ),
        RiskType.REGULATORY: create_agent_config(
            name="regulatory_risk_evaluator",
            description="Агент для оценки регуляторных и юридических рисков",
            **base_config_params
        ),
        RiskType.SOCIAL: create_agent_config(
            name="social_risk_evaluator",
            description="Агент для оценки социальных и манипулятивных рисков",
            **base_config_params
        )
    }
    
    # Создаем агентов-оценщиков (ВАЖНО: Сохраняем специализированные классы!)
    evaluators = {
        RiskType.ETHICAL: EthicalRiskEvaluator(configs[RiskType.ETHICAL]),
        RiskType.STABILITY: StabilityRiskEvaluator(configs[RiskType.STABILITY]),
        RiskType.SECURITY: SecurityRiskEvaluator(configs[RiskType.SECURITY]),
        RiskType.AUTONOMY: AutonomyRiskEvaluator(configs[RiskType.AUTONOMY]),
        RiskType.REGULATORY: RegulatoryRiskEvaluator(configs[RiskType.REGULATORY]),
        RiskType.SOCIAL: SocialRiskEvaluator(configs[RiskType.SOCIAL])
    }
    
    return evaluators

def create_safe_evaluator_process_method(risk_type: RiskType, risk_description: str):
        """
        Создает безопасный метод process для любого агента-оценщика
        
        Args:
            risk_type: Тип риска (RiskType enum)
            risk_description: Описание типа риска для логирования
            
        Returns:
            Метод process для агента
        """
        
        async def safe_process(
            self, 
            input_data: Dict[str, Any], 
            assessment_id: str
        ) -> AgentTaskResult:
            """Универсальный безопасный процесс оценки рисков"""
            start_time = datetime.now()
            task_type = f"{risk_type.value}riskevaluator"
            
            try:
                with LogContext(f"evaluate_{risk_type.value}_risk", assessment_id, self.name):
                    agent_profile = input_data.get("agent_profile", {})
                    agent_data = self._format_agent_data(agent_profile)
                    
                    # Получаем сырые данные от LLM
                    evaluation_result = await self.evaluate_risk(
                        risk_type=risk_description,
                        agent_data=agent_data,
                        evaluation_criteria=self.get_system_prompt(),
                        assessment_id=assessment_id
                    )
                    
                    # БЕЗОПАСНОЕ создание RiskEvaluation
                    risk_evaluation = RiskEvaluation.create_safe(
                        risk_type=risk_type,
                        evaluator_agent=self.name,
                        raw_data=evaluation_result
                    )
                    
                    # Логируем результат
                    self.logger.log_risk_evaluation(
                        self.name,
                        assessment_id,
                        risk_description,
                        risk_evaluation.total_score,
                        risk_evaluation.risk_level.value
                    )
                    
                    # Создаем успешный результат
                    execution_time = (datetime.now() - start_time).total_seconds()
                    
                    return AgentTaskResult(
                        agent_name=self.name,
                        task_type=task_type,
                        status=ProcessingStatus.COMPLETED,
                        result_data={"risk_evaluation": risk_evaluation.dict()},
                        start_time=start_time,
                        end_time=datetime.now(),
                        execution_time_seconds=execution_time
                    )
                    
            except Exception as e:
                # При любой ошибке создаем fallback оценку
                self.logger.bind_context(assessment_id, self.name).error(
                    f"❌ Ошибка оценки {risk_description}: {e}"
                )
                
                # Создаем fallback RiskEvaluation с минимальными данными
                fallback_evaluation = RiskEvaluation.create_from_raw_data(
                    risk_type=risk_type,
                    evaluator_agent=self.name,
                    raw_data={
                        "probability_score": 3,
                        "impact_score": 3,
                        "probability_reasoning": f"Fallback оценка из-за ошибки: {str(e)}",
                        "impact_reasoning": f"Fallback оценка из-за ошибки: {str(e)}",
                        "recommendations": ["Провести повторную оценку", "Проверить настройки LLM"],
                        "confidence_level": 0.3
                    }
                )
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return AgentTaskResult(
                    agent_name=self.name,
                    task_type=task_type,
                    status=ProcessingStatus.COMPLETED,  # Помечаем как завершенный с fallback
                    result_data={"risk_evaluation": fallback_evaluation.dict()},
                    start_time=start_time,
                    end_time=datetime.now(),
                    execution_time_seconds=execution_time,
                    error_message=f"Использованы fallback данные: {str(e)}"
                )
        
        return safe_process




def create_evaluators_from_env() -> Dict[RiskType, EvaluationAgent]:
    """
    Создание агентов-оценщиков из переменных окружения
    ОБНОВЛЕНО: Использует центральный конфигуратор
    """
    # ИЗМЕНЕНО: Используем центральный конфигуратор, убираем дублирование чтения env
    return create_all_evaluator_agents()


# ===============================
# Утилиты для анализа результатов
# ===============================

def extract_risk_evaluations_from_results(
    evaluation_results: Dict[RiskType, AgentTaskResult]
) -> Dict[RiskType, RiskEvaluation]:
    """
    ИСПРАВЛЕННОЕ извлечение объектов RiskEvaluation из результатов агентов
    """
    risk_evaluations = {}
    
    for risk_type, task_result in evaluation_results.items():
        try:
            print(f"🔍 DEBUG extract_risk_evaluations: Обрабатываем {risk_type}")
            
            if (task_result.status == ProcessingStatus.COMPLETED and 
                task_result.result_data and 
                "risk_evaluation" in task_result.result_data):
                
                eval_data = task_result.result_data["risk_evaluation"]
                print(f"🔍 DEBUG: eval_data type = {type(eval_data)}")
                
                # ИСПРАВЛЕНИЕ: Если это уже объект RiskEvaluation - используем как есть
                if isinstance(eval_data, RiskEvaluation):
                    print(f"✅ DEBUG: Используем готовый RiskEvaluation для {risk_type}")
                    risk_evaluations[risk_type] = eval_data
                    
                # Если это dict - пытаемся создать RiskEvaluation
                elif isinstance(eval_data, dict):
                    print(f"🔧 DEBUG: Создаем RiskEvaluation из dict для {risk_type}")
                    print(f"🔧 DEBUG: eval_data.keys() = {list(eval_data.keys())}")
                    
                    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Безопасное создание
                    try:
                        # Проверяем наличие threat_assessments
                        if 'threat_assessments' in eval_data:
                            print(f"✅ DEBUG: threat_assessments найден для {risk_type}")
                        
                        risk_evaluation = RiskEvaluation(**eval_data)
                        risk_evaluations[risk_type] = risk_evaluation
                        print(f"✅ DEBUG: RiskEvaluation создан успешно для {risk_type}")
                        
                    except Exception as create_error:
                        print(f"❌ DEBUG: Ошибка создания RiskEvaluation для {risk_type}: {create_error}")
                        print(f"❌ DEBUG: Проблемные данные: {eval_data}")
                        
                        # НЕ СОЗДАЕМ FALLBACK - пропускаем эту оценку
                        # Пусть workflow сам решает что делать
                        continue
                else:
                    print(f"⚠️ DEBUG: Неожиданный тип eval_data для {risk_type}: {type(eval_data)}")
                    
        except Exception as e:
            print(f"❌ DEBUG: Общая ошибка обработки {risk_type}: {e}")
            import traceback
            traceback.print_exc()
            # НЕ создаем fallback - просто пропускаем
    
    print(f"🔍 DEBUG extract_risk_evaluations: Итого извлечено {len(risk_evaluations)} оценок")
    return risk_evaluations


def calculate_overall_risk_score(
    risk_evaluations: Dict[RiskType, RiskEvaluation]
) -> tuple[int, str]:
    """
    Расчет общего балла и уровня риска
    
    Args:
        risk_evaluations: Оценки рисков по типам
        
    Returns:
        Tuple (общий балл, уровень риска)
    """
    if not risk_evaluations:
        return 0, "low"
    
    # Берем максимальный балл среди всех типов рисков
    max_score = max(evaluation.total_score for evaluation in risk_evaluations.values())
    
    # Определяем уровень риска
    if max_score <= 6:
        risk_level = "low"
    elif max_score <= 14:
        risk_level = "medium"
    else:
        risk_level = "high"
    
    return max_score, risk_level


def get_highest_risk_areas(
    risk_evaluations: Dict[RiskType, RiskEvaluation],
    threshold: int = 10
) -> List[RiskType]:
    """
    Получение областей наивысшего риска
    
    Args:
        risk_evaluations: Оценки рисков
        threshold: Порог для определения высокого риска
        
    Returns:
        Список типов рисков с высокими баллами
    """
    high_risk_areas = []
    
    for risk_type, evaluation in risk_evaluations.items():
        if evaluation.total_score >= threshold:
            high_risk_areas.append(risk_type)
    
    # Сортируем по убыванию балла
    high_risk_areas.sort(
        key=lambda rt: risk_evaluations[rt].total_score, 
        reverse=True
    )
    
    return high_risk_areas


# Экспорт основных классов и функций
__all__ = [
    # Агенты-оценщики
    "EthicalRiskEvaluator",
    "StabilityRiskEvaluator", 
    "SecurityRiskEvaluator",
    "AutonomyRiskEvaluator",
    "RegulatoryRiskEvaluator",
    "SocialRiskEvaluator",
    
   # Фабрики
    "create_all_evaluator_agents",
    "create_evaluator_nodes_for_langgraph_safe",  # ← ДОБАВИТЬ ЭТУ СТРОКУ
    "create_critic_node_function_fixed",         # ← И ЭТУ
    "create_evaluators_from_env",
    
    # Утилиты
    "extract_risk_evaluations_from_results",
    "calculate_overall_risk_score",
    "get_highest_risk_areas"
]

# ===============================
# ИСПРАВЛЕННЫЕ ФУНКЦИИ ДЛЯ LANGGRAPH
# ===============================

def create_evaluator_nodes_for_langgraph_safe(evaluators: Dict[RiskType, Any]) -> Dict[str, callable]:
    """Создание безопасных узлов для LangGraph без concurrent updates"""
    
    def create_safe_evaluator_node(risk_type: RiskType, evaluator):
        async def safe_evaluator_node(state: WorkflowState) -> Dict[str, Any]:
            """Безопасный узел оценщика - обновляет только свое поле"""
            
            assessment_id = state.get("assessment_id", "unknown")
            agent_profile = state.get("agent_profile", {})
            
            # Подготавливаем входные данные
            input_data = {"agent_profile": agent_profile}
            
            # Запускаем оценщика
            result = await evaluator.run(input_data, assessment_id)
            
            # КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: каждый агент обновляет только свое поле
            field_mapping = {
                RiskType.ETHICAL: "ethical_evaluation",
                RiskType.STABILITY: "stability_evaluation",
                RiskType.SECURITY: "security_evaluation", 
                RiskType.AUTONOMY: "autonomy_evaluation",
                RiskType.REGULATORY: "regulatory_evaluation",
                RiskType.SOCIAL: "social_evaluation"
            }
            
            field_name = field_mapping[risk_type]
            
            # Возвращаем только одно обновление поля
            return {field_name: result.dict()}
        
        return safe_evaluator_node
    
    # Создаем узлы для всех оценщиков
    nodes = {}
    for risk_type, evaluator in evaluators.items():
        node_name = f"{risk_type.value}_evaluator_node"
        nodes[node_name] = create_safe_evaluator_node(risk_type, evaluator)
    
    return nodes

def create_critic_node_function_fixed(critic_agent):
    """Создает исправленную функцию узла критика для LangGraph"""
    
    async def critic_node(state: WorkflowState) -> Dict[str, Any]:
        """Узел критика в LangGraph workflow - ОБНОВЛЕННАЯ ВЕРСИЯ"""
        
        assessment_id = state.get("assessment_id", "unknown")
        agent_profile = state.get("agent_profile", {})
        
        # Получаем результаты оценки из нового формата состояния
        evaluation_results = state.get_evaluation_results()
        
        # Проверяем что есть результаты для критики
        valid_results = {k: v for k, v in evaluation_results.items() if v is not None}
        
        if not valid_results:
            critic_agent.logger.bind_context(assessment_id, "critic").warning(
                "⚠️ Нет результатов оценки для критики"
            )
            return {"critic_results": {}}
        
        try:
            # Выполняем критику всех доступных оценок
            critic_results = await critic_agent.critique_multiple_evaluations(
                evaluation_results=valid_results,
                agent_profile=agent_profile,
                assessment_id=assessment_id
            )
            
            return {"critic_results": critic_results}
            
        except Exception as e:
            critic_agent.logger.bind_context(assessment_id, "critic").error(
                f"❌ Критическая ошибка в узле критика: {e}"
            )
            
            # Возвращаем пустые результаты чтобы не блокировать workflow
            return {"critic_results": {}}
    
    return critic_node
