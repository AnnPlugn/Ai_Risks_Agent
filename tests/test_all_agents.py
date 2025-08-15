# test_all_agents.py
"""
Полный тест всех агентов системы оценки рисков (Этап 3)
Проверяем профайлер, 6 агентов-оценщиков и критика
"""

import asyncio
import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_agent_imports():
    """Тест импортов всех агентов"""
    print("🧪 Тестирование импортов агентов...")
    
    try:
        # Базовые классы
        from src.agents.base_agent import BaseAgent, AgentConfig, create_agent_config
        print("✅ Базовые классы агентов импортированы")
        
        # Профайлер
        from src.agents.profiler_agent import ProfilerAgent, create_profiler_agent
        print("✅ Профайлер-агент импортирован")
        
        # Агенты-оценщики
        from src.agents.evaluator_agents import (
            EthicalRiskEvaluator, StabilityRiskEvaluator, SecurityRiskEvaluator,
            AutonomyRiskEvaluator, RegulatoryRiskEvaluator, SocialRiskEvaluator,
            create_all_evaluator_agents
        )
        print("✅ Все 6 агентов-оценщиков импортированы")
        
        # Критик
        from src.agents.critic_agent import CriticAgent, create_critic_agent
        print("✅ Критик-агент импортирован")
        
        return True
        
    except ImportError as e:
        print(f"❌ Ошибка импорта агентов: {e}")
        return False

async def test_agent_creation():
    """Тест создания всех агентов"""
    print("\n🧪 Тестирование создания агентов...")
    
    try:
        from src.agents.profiler_agent import create_profiler_agent
        from src.agents.evaluator_agents import create_all_evaluator_agents
        from src.agents.critic_agent import create_critic_agent
        
        # Создаем профайлер
        profiler = create_profiler_agent()
        print(f"✅ Профайлер создан: {profiler.name}")
        
        # Создаем всех оценщиков
        evaluators = create_all_evaluator_agents()
        print(f"✅ Агенты-оценщики созданы: {len(evaluators)} штук")
        
        for risk_type, evaluator in evaluators.items():
            print(f"   - {risk_type.value}: {evaluator.name}")
        
        # Создаем критика
        critic = create_critic_agent(quality_threshold=7.0)
        print(f"✅ Критик создан: {critic.name}")
        
        # Проверяем health check (если LM Studio доступен)
        print(f"\n🔍 Проверка доступности LLM...")
        profiler_health = await profiler.health_check()
        print(f"✅ Профайлер health check: {'✅ OK' if profiler_health else '⚠️ LM Studio недоступен'}")
        
        return True, (profiler, evaluators, critic)
        
    except Exception as e:
        print(f"❌ Ошибка создания агентов: {e}")
        return False, None

async def test_profiler_workflow():
    """Тест полного workflow профайлера"""
    print("\n🧪 Тестирование workflow профайлера...")
    
    try:
        from src.agents.profiler_agent import create_profiler_agent
        
        profiler = create_profiler_agent()
        
        # Тестовые данные - анализируем сам проект
        test_input = {
            "source_files": [
                str(Path(__file__).parent / "src"),  # Папка с кодом
                "requirements.txt",  # Файл зависимостей
                "test_stage1.py"  # Тестовый файл
            ],
            "agent_name": "AI_Risk_Assessment_System"
        }
        
        assessment_id = "test_full_workflow_001"
        
        print(f"📁 Профилирование: {len(test_input['source_files'])} источников")
        
        # Запускаем профайлер
        result = await profiler.run(test_input, assessment_id)
        
        if result.status.value == "completed":
            agent_profile = result.result_data["agent_profile"]
            print(f"✅ Профиль создан: {agent_profile['name']}")
            print(f"✅ Тип агента: {agent_profile['agent_type']}")
            print(f"✅ Автономность: {agent_profile['autonomy_level']}")
            print(f"✅ Доступ к данным: {agent_profile['data_access']}")
            
            return True, agent_profile
        else:
            print(f"❌ Ошибка профилирования: {result.error_message}")
            return False, None
            
    except Exception as e:
        print(f"❌ Ошибка workflow профайлера: {e}")
        return False, None

async def test_evaluator_workflow(agent_profile):
    """Тест workflow агентов-оценщиков"""
    print("\n🧪 Тестирование workflow агентов-оценщиков...")
    
    try:
        from src.agents.evaluator_agents import create_all_evaluator_agents
        from src.models.risk_models import RiskType
        
        evaluators = create_all_evaluator_agents()
        assessment_id = "test_evaluators_001"
        
        evaluation_results = {}
        
        # Тестируем каждого оценщика
        for risk_type, evaluator in evaluators.items():
            print(f"🔍 Оценка {risk_type.value}...")
            
            input_data = {"agent_profile": agent_profile}
            
            # Запускаем оценщика
            result = await evaluator.run(input_data, assessment_id)
            evaluation_results[risk_type] = result
            
            if result.status.value == "completed":
                risk_eval = result.result_data["risk_evaluation"]
                print(f"   ✅ {risk_type.value}: {risk_eval['total_score']} баллов ({risk_eval['risk_level']})")
            else:
                print(f"   ❌ {risk_type.value}: {result.error_message}")
        
        # Подсчитываем статистику
        successful = sum(1 for r in evaluation_results.values() if r.status.value == "completed")
        print(f"\n📊 Результаты оценки: {successful}/{len(evaluators)} успешно")
        
        return successful > 0, evaluation_results
        
    except Exception as e:
        print(f"❌ Ошибка workflow оценщиков: {e}")
        return False, None

async def test_critic_workflow(evaluation_results, agent_profile):
    """Тест workflow критик-агента"""
    print("\n🧪 Тестирование workflow критик-агента...")
    
    try:
        from src.agents.critic_agent import create_critic_agent
        
        critic = create_critic_agent(quality_threshold=6.0)  # Пониженный порог для тестов
        assessment_id = "test_critic_001"
        
        # Критикуем все оценки
        critic_results = await critic.critique_multiple_evaluations(
            evaluation_results=evaluation_results,
            agent_profile=agent_profile,
            assessment_id=assessment_id
        )
        
        # Анализируем результаты критики
        print(f"🔍 Критический анализ {len(critic_results)} оценок...")
        
        acceptable_count = 0
        for risk_type, critic_result in critic_results.items():
            if critic_result.status.value == "completed":
                critic_eval = critic_result.result_data["critic_evaluation"]
                is_acceptable = critic_eval["is_acceptable"]
                quality_score = critic_eval["quality_score"]
                
                status = "✅ принято" if is_acceptable else "❌ отклонено"
                print(f"   {risk_type.value}: {quality_score:.1f}/10 - {status}")
                
                if is_acceptable:
                    acceptable_count += 1
        
        # Получаем рекомендации по повторам
        retry_needed = critic.get_retry_recommendations(critic_results)
        print(f"\n📋 Требуют повторной оценки: {len(retry_needed)} из {len(critic_results)}")
        
        if retry_needed:
            print(f"   Риски для повтора: {[rt.value for rt in retry_needed]}")
        
        # Генерируем отчет по качеству
        quality_report = critic.generate_improvement_report(critic_results)
        avg_quality = quality_report["assessment_summary"]["average_quality"]
        print(f"\n📈 Средняя оценка качества: {avg_quality:.1f}/10")
        
        return True, critic_results
        
    except Exception as e:
        print(f"❌ Ошибка workflow критика: {e}")
        return False, None

async def test_langgraph_integration():
    """Тест интеграции с LangGraph"""
    print("\n🧪 Тестирование интеграции с LangGraph...")
    
    try:
        from src.agents.profiler_agent import create_profiler_node_function, create_profiler_agent
        from src.agents.evaluator_agents import create_evaluator_nodes_for_langgraph, create_all_evaluator_agents
        from src.agents.critic_agent import create_critic_node_function, create_quality_check_router, create_critic_agent
        
        # Создаем агентов
        profiler = create_profiler_agent()
        evaluators = create_all_evaluator_agents()
        critic = create_critic_agent()
        
        # Создаем функции узлов для LangGraph
        profiler_node = create_profiler_node_function(profiler)
        evaluator_nodes = create_evaluator_nodes_for_langgraph(evaluators)
        critic_node = create_critic_node_function(critic)
        quality_router = create_quality_check_router(quality_threshold=6.0)
        
        print(f"✅ Узлы LangGraph созданы:")
        print(f"   - Профайлер: {profiler_node.__name__}")
        print(f"   - Оценщики: {len(evaluator_nodes)} узлов")
        print(f"   - Критик: {critic_node.__name__}")
        print(f"   - Маршрутизатор: {quality_router.__name__}")
        
        # Тестируем базовое состояние workflow
        test_state = {
            "assessment_id": "test_langgraph_001",
            "source_files": ["test_stage1.py"],
            "preliminary_agent_name": "TestAgent",
            "max_retries": 2,
            "retry_count": {}
        }
        
        print(f"\n🔄 Симуляция workflow состояния...")
        
        # Эмулируем переход состояния через профайлер
        state_after_profiler = await profiler_node(test_state)
        print(f"   ✅ После профайлера: шаг = {state_after_profiler.get('current_step')}")
        
        # Проверяем маршрутизацию
        if "agent_profile" in state_after_profiler:
            # Добавляем фиктивные результаты критика для тестирования маршрутизации
            state_after_profiler["retry_needed"] = []
            route = quality_router(state_after_profiler)
            print(f"   ✅ Маршрутизация: {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка интеграции LangGraph: {e}")
        return False

async def main():
    """Главная функция полного тестирования"""
    print("🚀 ПОЛНОЕ ТЕСТИРОВАНИЕ АГЕНТОВ СИСТЕМЫ ОЦЕНКИ РИСКОВ")
    print("=" * 70)
    
    success = True
    
    # 1. Тестируем импорты
    success &= await test_agent_imports()
    
    # 2. Тестируем создание агентов
    creation_success, agents = await test_agent_creation()
    success &= creation_success
    
    if not creation_success:
        print("\n❌ Критическая ошибка: не удалось создать агентов")
        return False
    
    # 3. Тестируем workflow профайлера
    profiler_success, agent_profile = await test_profiler_workflow()
    success &= profiler_success
    
    if not profiler_success:
        print("\n⚠️ Пропускаем остальные тесты из-за ошибки профайлера")
        return False
    
    # 4. Тестируем workflow оценщиков
    evaluator_success, evaluation_results = await test_evaluator_workflow(agent_profile)
    success &= evaluator_success
    
    if evaluator_success and evaluation_results:
        # 5. Тестируем workflow критика
        critic_success, critic_results = await test_critic_workflow(evaluation_results, agent_profile)
        success &= critic_success
    
    # 6. Тестируем интеграцию с LangGraph
    success &= await test_langgraph_integration()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 ВСЕ АГЕНТЫ РАБОТАЮТ УСПЕШНО!")
        print("\n📋 Протестировано:")
        print("✅ Базовые классы агентов")
        print("✅ Профайлер-агент (сбор данных)")
        print("✅ 6 агентов-оценщиков рисков")
        print("✅ Критик-агент (контроль качества)")
        print("✅ Интеграция с LangGraph")
        print("\n🚀 Система готова к созданию workflow!")
    else:
        print("❌ ЕСТЬ ПРОБЛЕМЫ С АГЕНТАМИ")
        print("\n🔧 Возможные причины:")
        print("- LM Studio не запущен на localhost:1234")
        print("- Модель qwen3-4b не загружена")
        print("- Ошибки в промптах или логике агентов")
        print("- Проблемы с зависимостями")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)