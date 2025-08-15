# simple_test.py
"""
Простой тест с минимальными файлами для диагностики
"""

import asyncio
import tempfile
import os
from pathlib import Path

async def simple_test():
    """Простой тест с одним файлом"""
    
    print("🧪 ПРОСТОЙ ТЕСТ С МИНИМАЛЬНЫМИ ДАННЫМИ")
    print("=" * 50)
    
    # Создаем один простой файл без китайских символов
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write('''# simple_agent.py
class SimpleAgent:
    def __init__(self):
        self.name = "SimpleAgent"
        self.prompt = "You are a helpful assistant"
    
    def process(self, text):
        return f"Processed: {text}"
''')
        test_file = f.name
    
    try:
        # Импортируем workflow
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        from src.workflow import create_workflow_from_env
        
        print("✅ Импорты успешны")
        
        # Создаем workflow
        workflow = create_workflow_from_env()
        print("✅ Workflow создан")
        
        # Запускаем на одном файле
        print(f"📄 Тестовый файл: {test_file}")
        
        result = await workflow.run_assessment(
            source_files=[test_file],
            agent_name="SimpleTestAgent",
            assessment_id="simple_test_123"
        )
        
        print(f"\n📊 РЕЗУЛЬТАТ:")
        print(f"   Success: {result.get('success')}")
        print(f"   Assessment ID: {result.get('assessment_id')}")
        print(f"   Current Step: {result.get('current_step')}")
        print(f"   Processing Time: {result.get('processing_time', 0):.2f}с")
        
        # Проверяем БД
        from src.models.database import get_db_manager
        
        db = await get_db_manager()
        
        # Смотрим что в БД
        from sqlalchemy import text
        async with db.async_session() as session:
            result_count = await session.execute(text("SELECT COUNT(*) FROM risk_assessments"))
            count = result_count.scalar()
            
            print(f"\n🗄️ БАЗА ДАННЫХ:")
            print(f"   Записей в risk_assessments: {count}")
            
            if count > 0:
                latest = await session.execute(text("SELECT id, overall_risk_level FROM risk_assessments ORDER BY assessment_timestamp DESC LIMIT 1"))
                row = latest.first()
                if row:
                    print(f"   Последняя запись: {row[0][:8]}... - {row[1]}")
        
        await db.close()
        
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Удаляем тестовый файл
        try:
            os.unlink(test_file)
            print("🗑️ Тестовый файл удален")
        except:
            pass


if __name__ == "__main__":
    asyncio.run(simple_test())