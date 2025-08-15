# test_db_save.py
"""
Тест сохранения данных в БД
"""

import asyncio
import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.database import get_db_manager
from src.models.risk_models import RiskType, RiskLevel, AgentProfile, AgentRiskAssessment
from datetime import datetime

async def test_db_save():
    """Тестируем прямое сохранение в БД"""
    
    print("🧪 ТЕСТ СОХРАНЕНИЯ В БАЗУ ДАННЫХ")
    print("=" * 40)
    
    try:
        # Подключаемся к БД
        db = await get_db_manager()
        print("✅ Подключение к БД успешно")
        
        # Проверяем текущее состояние
        from sqlalchemy import text
        async with db.async_session() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM risk_assessments"))
            before_count = result.scalar()
            print(f"📊 Записей ДО теста: {before_count}")
        
        # Создаем тестовый профиль агента
        test_profile = AgentProfile(
            name="TestAgent",
            version="1.0",
            description="Тестовый агент для проверки БД",
            agent_type="chatbot",
            llm_model="qwen3-4b",
            autonomy_level="manual",
            target_audience="developers",
            analyzed_files=["test.py"]
        )
        
        print("✅ Тестовый профиль создан")
        
        # Сохраняем профиль
        profile_id = await db.save_agent_profile(test_profile)
        print(f"✅ Профиль сохранен с ID: {profile_id}")
        
        # Создаем тестовую оценку
        from src.models.risk_models import RiskEvaluation
        
        test_evaluation = RiskEvaluation(
            risk_type=RiskType.ETHICAL,
            evaluator_agent="test_evaluator",
            probability_score=3,
            impact_score=3,
            total_score=9,
            risk_level=RiskLevel.MEDIUM,
            probability_reasoning="Тестовое обоснование вероятности",
            impact_reasoning="Тестовое обоснование тяжести",
            key_factors=["test_factor1", "test_factor2"],
            recommendations=["test_recommendation1"],
            confidence_level=0.8
        )
        
        # Создаем итоговую оценку
        test_assessment = AgentRiskAssessment(
            assessment_id="test_db_save_123",
            agent_profile=test_profile,
            risk_evaluations={RiskType.ETHICAL: test_evaluation},
            overall_risk_score=9,
            overall_risk_level=RiskLevel.MEDIUM,
            highest_risk_areas=[RiskType.ETHICAL],
            priority_recommendations=["test_recommendation"],
            suggested_guardrails=["test_guardrail"],
            processing_time_seconds=1.0,
            quality_checks_passed=True
        )
        
        print("✅ Тестовая оценка создана")
        
        # Сохраняем оценку
        saved_id = await db.save_risk_assessment(test_assessment, profile_id)
        print(f"✅ Оценка сохранена с ID: {saved_id}")
        
        # Проверяем что сохранилось
        async with db.async_session() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM risk_assessments"))
            after_count = result.scalar()
            print(f"📊 Записей ПОСЛЕ теста: {after_count}")
            
            if after_count > before_count:
                print("🎉 УСПЕХ! Данные сохранились в БД")
                
                # Пробуем прочитать
                latest = await session.execute(text("SELECT id, overall_risk_level, overall_risk_score FROM risk_assessments ORDER BY assessment_timestamp DESC LIMIT 1"))
                row = latest.first()
                if row:
                    print(f"📋 Последняя запись: {row[0][:8]}... | {row[1]} | {row[2]} баллов")
            else:
                print("❌ ОШИБКА! Данные НЕ сохранились")
        
        await db.close()
        
        return after_count > before_count
        
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_workflow_finalization():
    """Тестируем финализацию workflow"""
    
    print("\n🔧 ТЕСТ ФИНАЛИЗАЦИИ WORKFLOW")
    print("=" * 40)
    
    try:
        from src.workflow import create_workflow_from_env
        
        workflow = create_workflow_from_env()
        print("✅ Workflow создан")
        
        # Создаем простой тестовый файл
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write('''# test_agent.py
class TestAgent:
    def __init__(self):
        self.name = "Test"
        self.prompt = "You are helpful"
''')
            test_file = f.name
        
        print(f"📄 Тестовый файл: {test_file}")
        
        # Запускаем workflow
        result = await workflow.run_assessment(
            source_files=[test_file],
            agent_name="WorkflowTestAgent",
            assessment_id="workflow_test_456"
        )
        
        print(f"📊 Workflow результат:")
        print(f"   Success: {result.get('success')}")
        print(f"   Step: {result.get('current_step')}")
        print(f"   Assessment ID: {result.get('assessment_id')}")
        
        # Проверяем БД после workflow
        db = await get_db_manager()
        from sqlalchemy import text
        async with db.async_session() as session:
            result_db = await session.execute(text("SELECT COUNT(*) FROM risk_assessments WHERE id LIKE 'workflow_test_%'"))
            count = result_db.scalar()
            print(f"   БД записей с workflow_test: {count}")
        
        await db.close()
        
        # Удаляем тестовый файл
        import os
        os.unlink(test_file)
        
        return count > 0
        
    except Exception as e:
        print(f"❌ ОШИБКА WORKFLOW: {e}")
        return False


async def main():
    """Главная функция тестирования"""
    
    print("🚀 ДИАГНОСТИКА ПРОБЛЕМ С СОХРАНЕНИЕМ В БД")
    print("=" * 60)
    
    # Тест 1: Прямое сохранение
    direct_save_ok = await test_db_save()
    
    # Тест 2: Workflow финализация
    workflow_save_ok = await test_workflow_finalization()
    
    print(f"\n📊 ИТОГИ ДИАГНОСТИКИ:")
    print(f"   Прямое сохранение: {'✅ Работает' if direct_save_ok else '❌ НЕ работает'}")
    print(f"   Workflow сохранение: {'✅ Работает' if workflow_save_ok else '❌ НЕ работает'}")
    
    if direct_save_ok and not workflow_save_ok:
        print("\n🔍 ВЫВОД: Проблема в финализации workflow, БД работает")
    elif not direct_save_ok:
        print("\n🔍 ВЫВОД: Проблема в самой БД или моделях данных")
    elif direct_save_ok and workflow_save_ok:
        print("\n🔍 ВЫВОД: Все работает! Проблема была в другом месте")
    else:
        print("\n🔍 ВЫВОД: Требуется дополнительная диагностика")


if __name__ == "__main__":
    asyncio.run(main())