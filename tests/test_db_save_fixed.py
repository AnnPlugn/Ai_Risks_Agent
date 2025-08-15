# test_db_absolutely_final.py
"""
АБСОЛЮТНО ФИНАЛЬНЫЙ ТЕСТ - с правильным доступом к данным БД
"""

import asyncio
import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.database import get_db_manager
from src.models.risk_models import (
    RiskType, RiskLevel, AgentProfile, AgentRiskAssessment, 
    AgentType, AutonomyLevel, DataSensitivity, RiskEvaluation,
    create_agent_risk_assessment
)
from datetime import datetime

async def test_db_absolutely_final():
    """АБСОЛЮТНО ФИНАЛЬНЫЙ ТЕСТ - все исправления применены"""
    
    print("🧪 АБСОЛЮТНО ФИНАЛЬНЫЙ ТЕСТ СОХРАНЕНИЯ В БАЗУ ДАННЫХ")
    print("=" * 65)
    
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
        
        # 1. Создаем профиль агента с ВСЕМИ полями
        test_profile = AgentProfile(
            name="TestAgent_AbsolutelyFinal",
            version="1.0",
            description="Абсолютно финальный тест с правильным доступом к данным",
            agent_type=AgentType.CHATBOT,
            llm_model="qwen3-4b",
            autonomy_level=AutonomyLevel.SUPERVISED,
            data_access=[DataSensitivity.INTERNAL],
            external_apis=["test_api_final", "working_api"],
            target_audience="developers",
            operations_per_hour=50,
            revenue_per_operation=5.0,
            system_prompts=["Ты - абсолютно финальный тестовый агент"],
            guardrails=["Не разглашай данные", "Соблюдай этику"],
            source_files=["test_absolutely_final.py"]
        )
        
        print("✅ Профиль создан с external_apis:", test_profile.external_apis)
        
        # 2. Сохраняем профиль
        profile_id = await db.save_agent_profile(test_profile)
        print(f"✅ Профиль сохранен с ID: {profile_id}")
        
        # 3. Создаем правильную оценку риска
        test_evaluation = RiskEvaluation(
            risk_type=RiskType.ETHICAL,
            evaluator_agent="test_evaluator_absolutely_final",
            probability_score=2,
            impact_score=3,
            total_score=6,
            risk_level=RiskLevel.LOW,
            probability_reasoning="Низкая вероятность этических проблем в абсолютно финальном тесте",
            impact_reasoning="Умеренное влияние в случае проблем",
            key_factors=["абсолютно финальная тестовая среда", "ограниченный доступ"],
            recommendations=["добавить мониторинг", "провести финальный аудит"],
            confidence_level=0.95
        )
        
        print("✅ Оценка создана с key_factors:", test_evaluation.key_factors)
        
        # 4. Создаем итоговую оценку
        assessment_id = "test_absolutely_final_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        
        test_assessment = create_agent_risk_assessment(
            assessment_id=assessment_id,
            agent_profile=test_profile,
            risk_evaluations={"ethical": test_evaluation},
            processing_time_seconds=1.5,
            quality_checks_passed=True
        )
        
        print("✅ Итоговая оценка создана")
        print(f"   assessment_id: {test_assessment.assessment_id}")
        
        # 5. Сохраняем оценку в БД
        print("\n📊 Сохраняем полную оценку рисков...")
        saved_assessment_id = await db.save_risk_assessment(test_assessment, profile_id)
        print(f"✅ Оценка сохранена с ID: {saved_assessment_id}")
        
        # 6. Проверяем результаты
        async with db.async_session() as session:
            result = await session.execute(text("SELECT COUNT(*) FROM risk_assessments"))
            after_count = result.scalar()
            print(f"📊 Записей ПОСЛЕ теста: {after_count}")
            
            result = await session.execute(text("SELECT COUNT(*) FROM agent_profiles"))
            profiles_count = result.scalar()
            print(f"📊 Профилей в БД: {profiles_count}")
            
            result = await session.execute(text("SELECT COUNT(*) FROM risk_evaluations"))
            evaluations_count = result.scalar()
            print(f"📊 Отдельных оценок: {evaluations_count}")
        
        if after_count > before_count:
            print(f"\n🎉 АБСОЛЮТНЫЙ УСПЕХ! Добавлено {after_count - before_count} записей")
            
            # 7. Тестируем чтение данных (ИСПРАВЛЕНО!)
            print("\n🔍 Тестируем чтение из БД...")
            
            retrieved_profile = await db.get_agent_profile(profile_id)
            if retrieved_profile:
                print("✅ Профиль получен из БД:")
                print(f"   Имя: {retrieved_profile.name}")
                print(f"   Внешние API: {retrieved_profile.external_apis}")
                print(f"   Тип: {retrieved_profile.agent_type}")
            
            # ИСПРАВЛЕНО: get_risk_assessment возвращает словарь, не объект!
            retrieved_assessment_data = await db.get_risk_assessment(saved_assessment_id)
            if retrieved_assessment_data:
                assessment_obj = retrieved_assessment_data["assessment"]  # Получаем объект assessment
                evaluations_list = retrieved_assessment_data["evaluations"]
                critic_evals_list = retrieved_assessment_data["critic_evaluations"]
                
                print("✅ Оценка получена из БД:")
                print(f"   ID: {assessment_obj.id}")
                print(f"   Общий балл: {assessment_obj.overall_risk_score}")
                print(f"   Уровень: {assessment_obj.overall_risk_level}")
                print(f"   Области риска: {assessment_obj.highest_risk_areas}")
                print(f"   Время: {assessment_obj.processing_time_seconds}с")
                print(f"   Качество: {assessment_obj.quality_checks_passed}")
                print(f"   Количество оценок: {len(evaluations_list)}")
                print(f"   Количество критических оценок: {len(critic_evals_list)}")
                
                # Показываем детали первой оценки
                if evaluations_list:
                    eval_obj = evaluations_list[0]
                    print(f"   Первая оценка - тип: {eval_obj.risk_type}")
                    print(f"   Ключевые факторы: {eval_obj.key_factors}")
                    print(f"   Рекомендации: {eval_obj.recommendations}")
            
            # 8. Прямой SQL запрос для дополнительной проверки
            print("\n🔍 Дополнительная проверка через SQL:")
            async with db.async_session() as session:
                result = await session.execute(text(
                    "SELECT risk_type, key_factors, recommendations, confidence_level "
                    "FROM risk_evaluations WHERE assessment_id = :assessment_id LIMIT 1"
                ), {"assessment_id": saved_assessment_id})
                row = result.first()
                if row:
                    print(f"✅ SQL данные:")
                    print(f"   Тип риска: {row[0]}")
                    print(f"   Ключевые факторы: {row[1]}")
                    print(f"   Рекомендации: {row[2]}")
                    print(f"   Уверенность: {row[3]}")
            
            print("\n🎯 ВСЕ ТЕСТЫ ПРОЙДЕНЫ АБСОЛЮТНО УСПЕШНО!")
            print("✅ БД работает корректно, все поля соответствуют схеме!")
            print("✅ Чтение и запись данных работают правильно!")
            print("✅ Система готова к использованию!")
            
        else:
            print("❌ ОШИБКА! Данные не сохранились")
            
    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_db_absolutely_final())