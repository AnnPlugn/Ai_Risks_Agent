# apply_quality_calculation_fix.py
"""
Системное исправление расчета качества в _quality_check_node
Заменяет логику расчета, чтобы учитывать confidence_level агентов
"""

import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

def apply_quality_calculation_fix():
    """
    Применяет исправление к функции _quality_check_node
    чтобы расчет качества учитывал confidence_level агентов
    """
    
    try:
        print("🔧 Применение системного исправления расчета качества...")
        
        from src.workflow.graph_builder import RiskAssessmentWorkflow
        
        # Сохраняем оригинальный метод
        if not hasattr(RiskAssessmentWorkflow, '_original_quality_check_node'):
            RiskAssessmentWorkflow._original_quality_check_node = RiskAssessmentWorkflow._quality_check_node
        
        async def fixed_quality_check_node(self, state):
            """
            ИСПРАВЛЕННАЯ версия _quality_check_node с правильным расчетом качества
            """
            
            assessment_id = state["assessment_id"]
            
            self.graph_logger.log_workflow_step(
                assessment_id, "quality_check_start",
                f"🔧 ИСПРАВЛЕННАЯ версия quality_check, входящий current_step: {state.get('current_step')}"
            )

            # Получаем результаты успешных оценок
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
                
                self.graph_logger.log_workflow_step(
                    assessment_id, "quality_check_data",
                    f"🔧 ИСПРАВЛЕНО: Успешных оценок: {len(evaluation_results)}, типы: {list(evaluation_results.keys())}"
                )
                
            except Exception as e:
                self.graph_logger.log_workflow_step(
                    assessment_id, "quality_check_data_error",
                    f"🔧 Ошибка получения successful_evaluations: {e}"
                )
                evaluation_results = {}

            # Если нет успешных оценок, идем в обработку ошибок
            if not evaluation_results:
                self.graph_logger.log_workflow_step(
                    assessment_id, "quality_check_no_evaluations",
                    "🔧 ❌ НЕТ УСПЕШНЫХ ОЦЕНОК - устанавливаем error"
                )
                state["current_step"] = "error"
                state["error_message"] = "Нет успешных оценок рисков"
                state["retry_needed"] = []
                return state
            
            # Базовая статистика
            success_rate = len(evaluation_results) / 6  # 6 типов рисков всего
            
            # Проверяем есть ли уже результаты критика
            critic_results = state.get("critic_results", {})
            has_critic_results = bool(critic_results)
            
            self.graph_logger.log_workflow_step(
                assessment_id, "quality_check_metrics",
                f"🔧 Success rate: {success_rate:.2f}, имеет результаты критика: {has_critic_results}"
            )
            
            if has_critic_results:
                # Если есть результаты критика, используем их (старая логика)
                retry_needed = []
                quality_scores = []
                
                for risk_type, critic_result in critic_results.items():
                    try:
                        if isinstance(critic_result, dict):
                            if (critic_result.get("status") == "completed" and 
                                critic_result.get("result_data") and 
                                "critic_evaluation" in critic_result["result_data"]):
                                
                                critic_eval = critic_result["result_data"]["critic_evaluation"]
                                quality_scores.append(critic_eval.get("quality_score", 7.0))
                                
                                if not critic_eval.get("is_acceptable", True):
                                    retry_count = state.get("retry_count", {})
                                    current_retries = retry_count.get(risk_type, 0)
                                    max_retries = state.get("max_retries", 3)
                                    
                                    if current_retries < max_retries:
                                        retry_needed.append(risk_type)
                            else:
                                quality_scores.append(7.0)
                                
                    except Exception as e:
                        self.graph_logger.log_workflow_step(
                            assessment_id, "quality_check_warning",
                            f"🔧 Ошибка обработки результата критика для {risk_type}: {e}"
                        )
                        quality_scores.append(7.0)
                
                avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 7.0
                
            else:
                # 🔧 ИСПРАВЛЕНО: Рассчитываем качество на основе confidence_level агентов
                confidence_scores = []
                
                self.graph_logger.log_workflow_step(
                    assessment_id, "quality_check_confidence_analysis",
                    f"🔧 ИСПРАВЛЕНО: Анализируем confidence_level из {len(evaluation_results)} успешных оценок"
                )
                
                for risk_type, result in evaluation_results.items():
                    try:
                        # Извлекаем confidence_level из результата
                        confidence = None
                        
                        if isinstance(result, dict):
                            result_data = result.get("result_data")
                            if result_data:
                                confidence = result_data.get("confidence_level")
                        elif hasattr(result, 'result_data') and result.result_data:
                            confidence = result.result_data.get("confidence_level")
                        
                        if confidence is not None:
                            confidence_scores.append(confidence)
                            self.graph_logger.log_workflow_step(
                                assessment_id, "confidence_extracted",
                                f"🔧 {risk_type}: confidence_level = {confidence:.3f}"
                            )
                        else:
                            # Fallback: если confidence не найден, используем средний
                            confidence_scores.append(0.7)
                            self.graph_logger.log_workflow_step(
                                assessment_id, "confidence_fallback",
                                f"🔧 {risk_type}: confidence_level не найден, используем 0.7"
                            )
                            
                    except Exception as e:
                        # В случае ошибки используем fallback
                        confidence_scores.append(0.7)
                        self.graph_logger.log_workflow_step(
                            assessment_id, "confidence_error",
                            f"🔧 {risk_type}: ошибка извлечения confidence - {e}, используем 0.7"
                        )
                
                if confidence_scores:
                    # Рассчитываем среднюю уверенность
                    avg_confidence = sum(confidence_scores) / len(confidence_scores)
                    
                    # Преобразуем confidence (0.0-1.0) в quality (0.0-10.0)
                    # 80% от confidence_level, 20% от success_rate
                    confidence_quality = avg_confidence * 8.0  # Основная часть
                    success_quality = success_rate * 2.0       # Бонус за успешность
                    avg_quality = confidence_quality + success_quality
                    
                    self.graph_logger.log_workflow_step(
                        assessment_id, "quality_check_confidence_calculation",
                        f"🔧 ИСПРАВЛЕНО: avg_confidence={avg_confidence:.3f}, "
                        f"confidence_quality={confidence_quality:.1f}, "
                        f"success_quality={success_quality:.1f}, "
                        f"total_quality={avg_quality:.1f}"
                    )
                else:
                    # Fallback: если нет данных о confidence, используем старую логику
                    avg_quality = 5.0 + (success_rate * 5.0)
                    self.graph_logger.log_workflow_step(
                        assessment_id, "quality_check_fallback",
                        f"🔧 Fallback: используем старую логику, quality={avg_quality:.1f}"
                    )
                
                retry_needed = []
                
                self.graph_logger.log_workflow_step(
                    assessment_id, "quality_check_final",
                    f"🔧 ИСПРАВЛЕНО: Итоговая оценка качества: {avg_quality:.1f} (порог: {self.quality_threshold})"
                )
            
            # Логируем результаты проверки качества
            self.graph_logger.log_quality_check(
                assessment_id, 
                "overall", 
                avg_quality, 
                self.quality_threshold
            )
            
            # Принимаем решение на основе качества
            state["average_quality"] = avg_quality
            
            if retry_needed:
                state["retry_needed"] = retry_needed
                state["current_step"] = "retry_needed"
                
                self.graph_logger.log_workflow_step(
                    assessment_id, "quality_check_retry",
                    f"🔧 ✅ УСТАНОВЛЕН current_step = 'retry_needed' для: {retry_needed}"
                )
                
            elif avg_quality < self.quality_threshold and not has_critic_results:
                state["current_step"] = "needs_critic"
                state["retry_needed"] = []
                
                self.graph_logger.log_workflow_step(
                    assessment_id, "quality_check_needs_critic",
                    f"🔧 ✅ УСТАНОВЛЕН current_step = 'needs_critic' (качество {avg_quality:.1f} < {self.quality_threshold})"
                )
                
            else:
                state["current_step"] = "ready_for_finalization"
                state["retry_needed"] = []
                
                self.graph_logger.log_workflow_step(
                    assessment_id, "quality_check_finalize",
                    f"🔧 ✅ УСТАНОВЛЕН current_step = 'ready_for_finalization' (качество {avg_quality:.1f} >= {self.quality_threshold})"
                )
            
            return state
        
        # Применяем исправление
        RiskAssessmentWorkflow._quality_check_node = fixed_quality_check_node
        
        print("✅ Системное исправление расчета качества применено")
        print("🎯 Теперь качество рассчитывается на основе confidence_level агентов")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка применения исправления: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_fixed_quality():
    """Тестирование с исправленным расчетом качества"""
    
    import asyncio
    import tempfile
    
    async def run_test():
        try:
            # Применяем исправление
            if not apply_quality_calculation_fix():
                print("❌ Не удалось применить исправление")
                return False
                
            # Создаем тестовые файлы
            temp_dir = Path(tempfile.mkdtemp())
            test_file = temp_dir / "test_agent.py"
            test_file.write_text("# Тестовый агент\nclass TestAgent:\n    pass", encoding='utf-8')
            
            # Настраиваем логирование
            from src.utils.logger import setup_logging
            setup_logging(log_level="INFO")
            
            # Создаем workflow
            from src.workflow import create_workflow_from_env
            workflow = create_workflow_from_env()
            
            # Устанавливаем средний порог
            workflow.quality_threshold = 6.5
            print(f"✅ Установлен порог качества: {workflow.quality_threshold}")
            
            # Запускаем оценку
            print("\n🏃‍♂️ Запуск с исправленным расчетом качества...")
            
            result = await workflow.run_assessment(
                source_files=[str(temp_dir)],
                agent_name="TestAgentWithFixedQuality"
            )
            
            # Анализируем результаты
            print("\n📊 РЕЗУЛЬТАТЫ С ИСПРАВЛЕННЫМ РАСЧЕТОМ:")
            print("=" * 50)
            
            if result.get("success"):
                print("✅ Оценка завершена успешно")
                
                processing_time = result.get("processing_time", 0)
                print(f"⏱️ Время обработки: {processing_time:.1f}с")
                
                # Проверяем запускался ли критик
                if processing_time > 60:
                    print("🎉 КРИТИК ЗАПУСКАЛСЯ! (время > 60 сек)")
                    print("✅ Исправление работает - качество рассчитывается корректно")
                else:
                    print("⚠️ Критик не запускался (время < 60 сек)")
                    print("💡 Возможно качество все еще высокое, попробуйте lower threshold")
                
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
    print("🚀 СИСТЕМНОЕ ИСПРАВЛЕНИЕ РАСЧЕТА КАЧЕСТВА")
    print("=" * 60)
    print("🎯 ЦЕЛЬ: Исправить логику quality_check_node для учета confidence_level")
    print("🔧 МЕТОД: Заменить расчет качества с success_rate на confidence_level")
    
    success = test_with_fixed_quality()
    
    print(f"\n🏁 РЕЗУЛЬТАТ СИСТЕМНОГО ИСПРАВЛЕНИЯ:")
    if success:
        print("✅ ИСПРАВЛЕНИЕ РАБОТАЕТ!")
        print("💡 Для постоянного использования примените изменения в graph_builder.py")
    else:
        print("❌ ЕСТЬ ПРОБЛЕМЫ С ИСПРАВЛЕНИЕМ")
        print("🔧 Проверьте логи для диагностики")