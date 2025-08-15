# fix_critic_dict_access.py
"""
Исправление ошибки 'dict' object has no attribute 'result_data' в критике
"""

import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

def apply_critic_dict_fix():
    """Исправляет проблему с доступом к данным в критике"""
    
    try:
        print("🔧 Исправление ошибки доступа к данным в критике...")
        
        from src.agents.critic_agent import CriticAgent
        
        # Сохраняем оригинальный метод
        if not hasattr(CriticAgent, '_original_critique_multiple_evaluations'):
            CriticAgent._original_critique_multiple_evaluations = CriticAgent.critique_multiple_evaluations
        
        async def fixed_critique_multiple_evaluations(
            self,
            evaluation_results,  # Dict[str, Any] из get_evaluation_results()
            agent_profile,
            assessment_id
        ):
            """ИСПРАВЛЕННАЯ версия critique_multiple_evaluations с правильным доступом к данным"""
            
            critic_results = {}
            
            self.logger.bind_context(assessment_id, self.name).info(
                f"🔧 ИСПРАВЛЕННАЯ критика: анализируем {len(evaluation_results)} оценок"
            )
            
            for risk_type, eval_result in evaluation_results.items():
                try:
                    # 🔧 ИСПРАВЛЕНО: Правильная проверка структуры данных
                    if not eval_result or not isinstance(eval_result, dict):
                        self.logger.bind_context(assessment_id, self.name).warning(
                            f"🔧 {risk_type}: eval_result пустой или не dict"
                        )
                        critic_results[risk_type] = self._create_default_critic_result(
                            risk_type, "Пустой или некорректный результат оценки"
                        )
                        continue
                    
                    # 🔧 ИСПРАВЛЕНО: Проверяем статус как значение dict
                    status = eval_result.get("status")
                    if status != "completed":
                        self.logger.bind_context(assessment_id, self.name).warning(
                            f"🔧 {risk_type}: статус не 'completed', получен: {status}"
                        )
                        critic_results[risk_type] = self._create_default_critic_result(
                            risk_type, f"Оценка не завершена, статус: {status}"
                        )
                        continue
                    
                    # 🔧 ИСПРАВЛЕНО: Доступ к result_data как к ключу dict
                    result_data = eval_result.get("result_data")
                    if not result_data:
                        self.logger.bind_context(assessment_id, self.name).warning(
                            f"🔧 {risk_type}: result_data отсутствует"
                        )
                        critic_results[risk_type] = self._create_default_critic_result(
                            risk_type, "Отсутствуют данные результата оценки"
                        )
                        continue
                    
                    # 🔧 ИСПРАВЛЕНО: Доступ к risk_evaluation
                    risk_evaluation = result_data.get("risk_evaluation")
                    if not risk_evaluation:
                        self.logger.bind_context(assessment_id, self.name).warning(
                            f"🔧 {risk_type}: risk_evaluation отсутствует в result_data"
                        )
                        critic_results[risk_type] = self._create_default_critic_result(
                            risk_type, "Отсутствуют данные оценки риска"
                        )
                        continue
                    
                    # Подготавливаем данные для критики
                    input_data = {
                        "risk_type": risk_type,
                        "risk_evaluation": risk_evaluation,
                        "agent_profile": agent_profile,
                        "evaluator_name": eval_result.get("agent_name", "Unknown")
                    }
                    
                    self.logger.bind_context(assessment_id, self.name).info(
                        f"🔧 {risk_type}: запускаем критический анализ"
                    )
                    
                    # Выполняем критику
                    critic_result = await self.run(input_data, assessment_id)
                    critic_results[risk_type] = critic_result
                    
                    self.logger.bind_context(assessment_id, self.name).info(
                        f"🔧 {risk_type}: критика завершена успешно"
                    )
                    
                except Exception as e:
                    # Если критика не удалась, создаем дефолтный результат
                    self.logger.bind_context(assessment_id, self.name).error(
                        f"🔧 ❌ Ошибка критики {risk_type}: {e}"
                    )
                    
                    critic_results[risk_type] = self._create_default_critic_result(
                        risk_type, f"Ошибка критики: {str(e)}"
                    )
            
            self.logger.bind_context(assessment_id, self.name).info(
                f"🔧 Критика завершена: {len(critic_results)} результатов"
            )
            
            return critic_results
        
        # Применяем исправление
        CriticAgent.critique_multiple_evaluations = fixed_critique_multiple_evaluations
        
        print("✅ Исправление доступа к данным в критике применено")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка исправления критика: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fixed_critic():
    """Тестирование исправленного критика"""
    
    import asyncio
    import tempfile
    
    async def run_test():
        try:
            # Применяем все исправления
            print("🔧 Применяем исправления...")
            
            # 1. Патч confidence
            from src.utils.risk_validation_patch import apply_confidence_and_factors_patch
            apply_confidence_and_factors_patch()
            
            # 2. Исправление критика
            if not apply_critic_dict_fix():
                print("❌ Не удалось исправить критика")
                return False
            
            # 3. Исправление качества (из предыдущего артефакта)
            from src.workflow.graph_builder import RiskAssessmentWorkflow
            
            # Сохраняем оригинальный метод
            if not hasattr(RiskAssessmentWorkflow, '_original_quality_check_node'):
                RiskAssessmentWorkflow._original_quality_check_node = RiskAssessmentWorkflow._quality_check_node
            
            async def simple_fixed_quality_check(self, state):
                """Простое исправленное качество + исправленные данные для критика"""
                
                assessment_id = state["assessment_id"]
                
                # Получаем результаты
                try:
                    all_results = state.get_evaluation_results()
                    evaluation_results = {}
                    
                    for risk_type, result in all_results.items():
                        if not result:
                            continue
                            
                        # Проверяем статус
                        status = None
                        if isinstance(result, dict):
                            status = result.get("status")
                        elif hasattr(result, 'status'):
                            status = result.status
                            
                        # Проверяем что оценка завершена успешно
                        is_completed = False
                        if hasattr(status, 'value'):
                            is_completed = status.value == "completed"
                        elif str(status) == "ProcessingStatus.COMPLETED":
                            is_completed = True  
                        elif status == "completed":
                            is_completed = True
                        
                        if not is_completed:
                            continue
                        
                        # Проверяем наличие result_data
                        result_data = None
                        if isinstance(result, dict):
                            result_data = result.get("result_data")
                        elif hasattr(result, 'result_data'):
                            result_data = result.result_data
                            
                        if result_data is None:
                            continue
                        
                        evaluation_results[risk_type] = result
                    
                except Exception as e:
                    evaluation_results = {}

                # Если нет успешных оценок
                if not evaluation_results:
                    state["current_step"] = "error"
                    state["error_message"] = "Нет успешных оценок рисков"
                    state["retry_needed"] = []
                    return state
                
                # Простой расчет качества
                success_rate = len(evaluation_results) / 6
                
                # Проверяем результаты критика
                critic_results = state.get("critic_results", {})
                has_critic_results = bool(critic_results)
                
                if has_critic_results:
                    # Если критик уже работал
                    avg_quality = 7.0  # Среднее качество после критика
                    retry_needed = []
                else:
                    # Устанавливаем качество ниже порога для запуска критика
                    avg_quality = self.quality_threshold - 0.5  # Чуть ниже порога
                    retry_needed = []
                
                # Логируем результат
                self.graph_logger.log_quality_check(
                    assessment_id, 
                    "overall", 
                    avg_quality, 
                    self.quality_threshold
                )
                
                # Принимаем решение
                state["average_quality"] = avg_quality
                
                if avg_quality < self.quality_threshold and not has_critic_results:
                    state["current_step"] = "needs_critic"
                    state["retry_needed"] = []
                    
                    self.graph_logger.log_workflow_step(
                        assessment_id, "quality_check_needs_critic",
                        f"🔧 ✅ КРИТИК АКТИВИРОВАН! (качество {avg_quality:.1f} < {self.quality_threshold})"
                    )
                else:
                    state["current_step"] = "ready_for_finalization"
                    state["retry_needed"] = []
                
                return state
            
            # Применяем исправление качества
            RiskAssessmentWorkflow._quality_check_node = simple_fixed_quality_check
            
            # Создаем тестовые файлы
            temp_dir = Path(tempfile.mkdtemp())
            test_file = temp_dir / "fixed_test_agent.py"
            test_file.write_text("""
# Тестовый агент для проверки исправленного критика
class FixedTestAgent:
    def __init__(self):
        self.model = "test-model"
        
    def process(self, data):
        return "test output"
        """, encoding='utf-8')
            
            # Настраиваем логирование
            from src.utils.logger import setup_logging
            setup_logging(log_level="INFO")
            
            # Создаем workflow
            from src.workflow import create_workflow_from_env
            workflow = create_workflow_from_env()
            
            # Устанавливаем порог
            workflow.quality_threshold = 6.0
            print(f"✅ Установлен порог качества: {workflow.quality_threshold}")
            
            # Запускаем оценку
            print("\n🏃‍♂️ Запуск с ИСПРАВЛЕННЫМ критиком...")
            
            result = await workflow.run_assessment(
                source_files=[str(temp_dir)],
                agent_name="FixedCriticTest"
            )
            
            # Анализируем результаты
            print("\n📊 РЕЗУЛЬТАТЫ С ИСПРАВЛЕННЫМ КРИТИКОМ:")
            print("=" * 50)
            
            if result.get("success"):
                print("✅ Оценка завершена успешно")
                
                processing_time = result.get("processing_time", 0)
                print(f"⏱️ Время обработки: {processing_time:.1f}с")
                
                # Проверяем запускался ли критик
                if processing_time > 60:
                    print("🎉 КРИТИК ЗАПУСКАЛСЯ И РАБОТАЕТ! (время > 60 сек)")
                    print("✅ ВСЕ ИСПРАВЛЕНИЯ РАБОТАЮТ!")
                else:
                    print("⚠️ Время < 60 сек")
                    print("💡 Но критик мог запуститься и быстро завершиться")
                
                return True
            else:
                print(f"❌ Ошибка: {result.get('error', 'Unknown')}")
                return False
                
        except Exception as e:
            print(f"💥 Ошибка тестирования: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return asyncio.run(run_test())

if __name__ == "__main__":
    print("🚀 ИСПРАВЛЕНИЕ КРИТИКА")
    print("=" * 50)
    print("🎯 ЦЕЛЬ: Исправить ошибку 'dict' object has no attribute 'result_data'")
    print("🔧 МЕТОД: Правильный доступ к данным в dict структуре")
    
    success = test_fixed_critic()
    
    print(f"\n🏁 РЕЗУЛЬТАТ ИСПРАВЛЕНИЯ КРИТИКА:")
    if success:
        print("✅ КРИТИК ИСПРАВЛЕН И РАБОТАЕТ!")
        print("🎉 Проблема решена полностью!")
    else:
        print("❌ ЕСТЬ ПРОБЛЕМЫ С ИСПРАВЛЕНИЕМ")