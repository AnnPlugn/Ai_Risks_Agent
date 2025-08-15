"""
Тест исправленного workflow после всех критических патчей
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_completely_fixed_workflow():
    """Тест полностью исправленного workflow"""
    print("🧪 Тестирование полностью исправленного workflow...")
    
    try:
        from src.workflow import create_workflow_from_env
        from src.models.risk_models import WorkflowState
        
        print("✅ Импорты успешны")
        
        # Создаем workflow
        workflow = create_workflow_from_env()
        print("✅ Workflow создан с исправлениями")
        
        # Проверяем LLM
        health = await workflow.profiler.health_check()
        print(f"✅ LLM Health: {'OK' if health else 'Недоступен'}")
        
        # Создаем простой тестовый файл с четкой структурой
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
# simple_test_agent.py
"""
Простой тестовый агент для системы оценки рисков
"""

class SimpleTestAgent:
    def __init__(self):
        self.name = "SimpleTestAgent"
        self.version = "1.0"
        self.system_prompt = "Ты простой тестовый помощник для проверки системы"
        
        # Ограничения безопасности
        self.guardrails = [
            "Работать только с тестовыми данными",
            "Не обрабатывать реальную персональную информацию",
            "Требовать подтверждения для любых действий"
        ]
    
    def process_request(self, request: str) -> str:
        """Обработка простых запросов"""
        if "тест" in request.lower():
            return "Тестовый ответ: запрос обработан"
        return "Стандартный ответ от тестового агента"
    
    def get_capabilities(self):
        """Возвращает возможности агента"""
        return [
            "Обработка тестовых запросов",
            "Генерация тестовых ответов",
            "Валидация входных данных"
        ]
''')
            test_file = f.name
        
        # Создаем документацию
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('''Документация: SimpleTestAgent

ОБЩАЯ ИНФОРМАЦИЯ:
Название: SimpleTestAgent
Версия: 1.0
Тип: Тестовый чат-бот
Назначение: Проверка системы оценки рисков

ФУНКЦИОНАЛЬНОСТЬ:
- Обработка простых тестовых запросов
- Генерация стандартных ответов
- Валидация входных данных

ОГРАНИЧЕНИЯ БЕЗОПАСНОСТИ:
- Работает только с тестовыми данными
- Не обрабатывает персональную информацию
- Требует подтверждения для действий
- Под постоянным надзором

ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ:
- Язык: Python
- Автономность: Под надзором
- Доступ к данным: Только тестовые
- Интеграции: Отсутствуют
''')
            doc_file = f.name
        
        print("✅ Тестовые файлы созданы")
        
        try:
            # Запускаем workflow с исправлениями
            print("🔄 Запуск исправленного workflow...")
            
            result = await workflow.run_assessment(
                source_files=[test_file, doc_file],
                agent_name="SimpleTestAgent"
            )
            
            if result.get("success"):
                print("✅ Workflow завершен УСПЕШНО!")
                
                assessment = result.get("final_assessment")
                if assessment:
                    print(f"📊 Общий риск: {assessment.get('overall_risk_level', 'unknown')}")
                    print(f"🔢 Балл риска: {assessment.get('overall_risk_score', 0)}/25")
                    print(f"⏱️ Время обработки: {result.get('processing_time', 0):.1f}с")
                    
                    # Проверяем оценки рисков
                    risk_evaluations = assessment.get("risk_evaluations", {})
                    print(f"🔍 Получено оценок: {len(risk_evaluations)}/6")
                    
                    for risk_type, evaluation in risk_evaluations.items():
                        if isinstance(evaluation, dict):
                            level = evaluation.get("risk_level", "unknown")
                            score = evaluation.get("total_score", 0)
                            print(f"  • {risk_type}: {level} ({score}/25)")
                    
                    # Проверяем рекомендации
                    recommendations = assessment.get("priority_recommendations", [])
                    print(f"💡 Рекомендаций: {len(recommendations)}")
                    
                    return True
                else:
                    print("❌ Отсутствует итоговая оценка")
                    return False
            else:
                error = result.get("error", "Неизвестная ошибка")
                print(f"❌ Ошибка workflow: {error}")
                
                # Если ошибка связана с concurrent updates, это нормально
                if "INVALID_CONCURRENT_GRAPH_UPDATE" in error:
                    print("⚠️ Ошибка concurrent updates - требуется дополнительное исправление")
                
                return False
                
        finally:
            # Очищаем файлы
            try:
                import os
                os.unlink(test_file)
                os.unlink(doc_file)
                print("🗑️ Тестовые файлы удалены")
            except:
                pass
                
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Главная функция исправленного теста"""
    print("🚀 ТЕСТ ПОЛНОСТЬЮ ИСПРАВЛЕННОГО WORKFLOW")
    print("=" * 55)
    
    success = await test_completely_fixed_workflow()
    
    if success:
        print("\\n🎉 ВСЕ ИСПРАВЛЕНИЯ РАБОТАЮТ!")
        print("✅ Workflow полностью функционален")
        print("🚀 Система готова к использованию")
        
        print("\\n📝 Доступные команды:")
        print("   python main.py demo")
        print("   python main.py assess <файлы>")
        print("   python test_complete_workflow.py")
    else:
        print("\\n❌ ТРЕБУЮТСЯ ДОПОЛНИТЕЛЬНЫЕ ИСПРАВЛЕНИЯ")
        print("🔧 Проверьте логи выше для диагностики")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)