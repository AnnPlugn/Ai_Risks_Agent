# test_agents_quick_fixed.py
"""
Быстрый тест с исправлениями для qwen3-4b
"""

import asyncio
import sys
import time
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_single_evaluator_fixed():
    """Тест одного оценщика с исправлениями"""
    print("🧪 Тест оценщика этических рисков (исправленная версия)...")
    
    try:
        from src.agents.evaluator_agents import EthicalRiskEvaluator
        from src.agents.base_agent import create_agent_config
        
        # Простейший профиль для тестирования
        simple_profile = {
            "name": "QuickTestBot",
            "agent_type": "chatbot", 
            "description": "Простой бот для тестирования оценки рисков",
            "autonomy_level": "supervised",
            "data_access": ["public"],
            "target_audience": "Тестировщики системы",
            "llm_model": "qwen3-4b",
            "system_prompts": ["Ты помощник для тестирования"],
            "guardrails": ["Отвечай только на тестовые вопросы"],
            "external_apis": []
        }
        
        # Создаем оценщика с коротким тайм-аутом
        config = create_agent_config(
            name="ethical_evaluator_quick",
            description="Быстрая оценка этических рисков",
            llm_base_url="http://127.0.0.1:1234",
            llm_model="qwen3-4b", 
            temperature=0.0,  # Максимальная детерминированность
            timeout_seconds=45,  # Короткий тайм-аут
            use_risk_analysis_client=True
        )
        
        evaluator = EthicalRiskEvaluator(config)
        
        # Проверяем доступность LLM
        print("🔍 Проверка LLM...")
        is_available = await evaluator.health_check()
        if not is_available:
            print("❌ LM Studio недоступен")
            return False
        
        print("✅ LLM доступен, запускаем оценку...")
        
        input_data = {"agent_profile": simple_profile}
        assessment_id = "quick_eval_001"
        
        start_time = time.time()
        
        # Запускаем с таймаутом
        result = await asyncio.wait_for(
            evaluator.run(input_data, assessment_id),
            timeout=60  # 1 минута максимум
        )
        
        elapsed = time.time() - start_time
        print(f"⏱️ Время выполнения: {elapsed:.1f}с")
        
        if result.status.value == "completed":
            risk_eval = result.result_data["risk_evaluation"]
            print(f"✅ Успешная оценка!")
            print(f"   📊 Балл: {risk_eval['total_score']}/25")
            print(f"   📈 Уровень: {risk_eval['risk_level']}")
            print(f"   🎯 Вероятность: {risk_eval['probability_score']}/5")
            print(f"   💥 Воздействие: {risk_eval['impact_score']}/5")
            print(f"   🔑 Ключевые факторы: {len(risk_eval.get('key_factors', []))}")
            print(f"   💡 Рекомендации: {len(risk_eval.get('recommendations', []))}")
            return True
        else:
            print(f"❌ Ошибка оценки: {result.error_message}")
            return False
            
    except asyncio.TimeoutError:
        print("⏰ Тайм-аут! Оценка заняла больше 1 минуты")
        return False
    except Exception as e:
        print(f"❌ Исключение: {e}")
        return False

async def test_lightweight_profiler_fixed():
    """Тест профайлера с минимальными данными"""
    print("\n🧪 Тест профайлера (облегченная версия)...")
    
    try:
        from src.agents.profiler_agent import ProfilerAgent
        from src.agents.base_agent import create_agent_config
        
        # Создаем профайлер с коротким тайм-аутом
        config = create_agent_config(
            name="quick_profiler",
            description="Быстрый профайлер",
            llm_base_url="http://127.0.0.1:1234",
            llm_model="qwen3-4b",
            temperature=0.0,
            timeout_seconds=45,
            use_risk_analysis_client=False
        )
        
        profiler = ProfilerAgent(config)
        
        # Минимальные данные - создаем простой тестовый файл
        test_file = Path("quick_test_data.txt")
        test_content = """Тестовый агент: QuickBot
Описание: Простой чат-бот для демонстрации
Тип: Помощник
Функции: Отвечает на простые вопросы
Ограничения: Только тестовые данные"""
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        try:
            test_input = {
                "source_files": [str(test_file)],
                "agent_name": "QuickTestAgent"
            }
            
            print("🔄 Запуск быстрого профилирования...")
            start_time = time.time()
            
            result = await asyncio.wait_for(
                profiler.run(test_input, "quick_profile_001"),
                timeout=90  # 1.5 минуты максимум
            )
            
            elapsed = time.time() - start_time
            print(f"⏱️ Время выполнения: {elapsed:.1f}с")
            
            if result.status.value == "completed":
                agent_profile = result.result_data["agent_profile"]
                print(f"✅ Профиль создан!")
                print(f"   📝 Имя: {agent_profile['name']}")
                print(f"   🤖 Тип: {agent_profile['agent_type']}")
                print(f"   🎯 Аудитория: {agent_profile['target_audience']}")
                print(f"   🔒 Автономность: {agent_profile['autonomy_level']}")
                return True, agent_profile
            else:
                print(f"❌ Ошибка профилирования: {result.error_message}")
                return False, None
                
        finally:
            # Удаляем тестовый файл
            if test_file.exists():
                test_file.unlink()
            
    except asyncio.TimeoutError:
        print("⏰ Тайм-аут! Профилирование заняло больше 1.5 минут")
        return False, None
    except Exception as e:
        print(f"❌ Исключение: {e}")
        return False, None

async def main():
    """Главная функция быстрого тестирования с фиксами"""
    print("🚀 БЫСТРОЕ ТЕСТИРОВАНИЕ С ИСПРАВЛЕНИЯМИ")
    print("(оптимизировано для qwen3-4b)")
    print("=" * 50)
    
    total_start = time.time()
    
    # Тест 1: Быстрый оценщик
    evaluator_success = await test_single_evaluator_fixed()
    
    # Тест 2: Быстрый профайлер (только если оценщик работает)
    profiler_success = False
    if evaluator_success:
        profiler_success, profile = await test_lightweight_profiler_fixed()
    
    total_time = time.time() - total_start
    print(f"\n⏱️ Общее время: {total_time:.1f}с")
    
    print("\n" + "=" * 50)
    if evaluator_success and profiler_success:
        print("🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print("\n📋 Достижения:")
        print("✅ Агент-оценщик работает корректно")
        print("✅ Профайлер создает профили агентов")
        print("✅ JSON парсинг исправлен")
        print("✅ LLM интеграция функционирует")
        print("\n🚀 Система готова к созданию LangGraph workflow!")
    elif evaluator_success:
        print("🎯 ЧАСТИЧНЫЙ УСПЕХ!")
        print("✅ Агент-оценщик работает")
        print("⚠️ Профайлер нуждается в дополнительной оптимизации")
        print("\n💡 Можно переходить к Этапу 4 с рабочими оценщиками")
    else:
        print("❌ НУЖНЫ ДОПОЛНИТЕЛЬНЫЕ ИСПРАВЛЕНИЯ")
        print("\n🔧 Рекомендации:")
        print("1. Перезапустите LM Studio")
        print("2. Попробуйте модель поменьше (Phi-3-mini)")
        print("3. Или используйте mock-тестирование")
    
    return evaluator_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)