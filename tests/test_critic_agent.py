# test_critic_agent.py
"""
Тестовый скрипт для проверки работы агента-критика
Позволяет принудительно запустить критика с настраиваемыми параметрами
"""

import sys
import asyncio
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_agent_files() -> str:
    """Создание тестовых файлов агента для оценки"""
    print("📝 Создание тестовых файлов агента...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    # Создаем файл агента с умеренными рисками
    agent_file = temp_dir / "test_agent.py"
    agent_code = '''"""
Тестовый банковский ИИ-агент
"""

class BankingAssistant:
    def __init__(self):
        self.model = "gpt-4"
        self.temperature = 0.1
        self.system_prompt = """
        Ты - банковский ассистент. 
        Помогаешь клиентам с основными банковскими операциями.
        Ограничения:
        - Не разглашай конфиденциальную информацию
        - Требуй подтверждения для операций свыше 10,000 руб
        """
    
    def handle_request(self, user_input: str) -> str:
        """Обработка запроса пользователя"""
        # Базовая обработка без проверок безопасности
        if "баланс" in user_input.lower():
            return "Ваш баланс: 50,000 руб"
        elif "перевод" in user_input.lower():
            return "Перевод выполнен"
        else:
            return "Как могу помочь?"
    
    def access_customer_data(self, customer_id: str):
        """Доступ к данным клиента - потенциальный риск"""
        # ВНИМАНИЕ: нет проверки прав доступа
        return {"balance": 50000, "name": "Иван Иванов"}
'''
    
    agent_file.write_text(agent_code, encoding='utf-8')
    
    # Создаем файл конфигурации
    config_file = temp_dir / "config.json"
    config_data = {
        "agent_name": "TestBankingAssistant",
        "model": "gpt-4",
        "temperature": 0.1,
        "max_tokens": 2048,
        "system_prompt": "Ты банковский ассистент",
        "guardrails": [
            "Не разглашай персональные данные",
            "Требуй подтверждения для операций"
        ],
        "data_access": ["customer_profiles", "account_balances"],
        "autonomy_level": "semi_autonomous"
    }
    
    config_file.write_text(json.dumps(config_data, ensure_ascii=False, indent=2), encoding='utf-8')
    
    # Создаем документацию с описанием рисков
    doc_file = temp_dir / "agent_description.txt"
    doc_content = """Банковский ИИ-агент TestBankingAssistant

ОПИСАНИЕ:
Агент предназначен для помощи клиентам банка в выполнении базовых операций.
Имеет доступ к данным клиентов и может выполнять финансовые операции.

ФУНКЦИИ:
- Показ баланса счета
- Выполнение переводов между счетами
- Консультации по банковским продуктам
- Обработка жалоб клиентов

ТЕХНИЧЕСКИЕ ХАРАКТЕРИСТИКИ:
- Модель: GPT-4
- Температура: 0.1
- Максимум токенов: 2048
- Доступ к базе данных клиентов: ДА
- Возможность выполнения операций: ДА

МЕРЫ БЕЗОПАСНОСТИ:
- Требует подтверждение для операций свыше 10,000 руб
- Не должен разглашать персональные данные
- Ведет логи всех операций

ИЗВЕСТНЫЕ ПРОБЛЕМЫ:
- Отсутствует строгая аутентификация пользователей
- Нет проверки прав доступа к данным клиентов
- Недостаточный контроль над финансовыми операциями
"""
    
    doc_file.write_text(doc_content, encoding='utf-8')
    
    print(f"✅ Тестовые файлы созданы в: {temp_dir}")
    print(f"   - {agent_file.name}")
    print(f"   - {config_file.name}")
    print(f"   - {doc_file.name}")
    
    return str(temp_dir)

async def test_full_workflow_with_critic(
    agent_path: str, 
    quality_threshold: float = 5.0,
    force_critic: bool = True
) -> Dict[str, Any]:
    """Тестирование полного workflow с принудительным запуском критика"""
    
    print(f"\n🧪 Тестирование workflow с порогом качества: {quality_threshold}")
    print(f"🔍 Принудительный запуск критика: {force_critic}")
    
    try:
        from src.workflow import create_workflow_from_env
        from src.utils.logger import setup_logging
        
        # Настраиваем логирование
        setup_logging()
        
        # Создаем workflow
        workflow = create_workflow_from_env()
        
        # ВАЖНО: Устанавливаем низкий порог качества для гарантированного запуска критика
        if force_critic:
            workflow.quality_threshold = quality_threshold
            print(f"✅ Установлен порог качества: {workflow.quality_threshold}")
        
        # Запускаем оценку
        result = await workflow.run_assessment(
            source_files=[agent_path],
            agent_name="TestBankingAssistant"
        )
        
        return result
        
    except Exception as e:
        print(f"❌ Ошибка в workflow: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

async def test_critic_agent_directly():
    """Прямое тестирование агента-критика"""
    print("\n🤖 Прямое тестирование агента-критика...")
    
    try:
        from src.agents.critic_agent import create_critic_agent
        from src.models.risk_models import RiskType
        
        # Создаем критика с низким порогом качества
        critic = create_critic_agent(quality_threshold=5.0)
        print(f"✅ Критик создан с порогом качества: {critic.quality_threshold}")
        
        # Создаем тестовую оценку с проблемами качества
        test_evaluation = {
            "probability_score": 3,
            "impact_score": 3,
            "total_score": 9,
            "risk_level": "medium",
            "probability_reasoning": "Короткое обоснование",  # Плохое качество
            "impact_reasoning": "Очень краткое объяснение",  # Плохое качество  
            "key_factors": ["общие факторы"],  # Неконкретно
            "recommendations": ["улучшить безопасность"],  # Слишком общее
            "confidence_level": 0.6
        }
        
        # Тестовые данные агента
        agent_data = {
            "agent_name": "TestBankingAssistant",
            "description": "Банковский ассистент",
            "technical_specs": {"model": "gpt-4"},
            "data_access": ["customer_data"],
            "autonomy_level": "semi_autonomous"
        }
        
        print("🔍 Запуск критической оценки...")
        
        # Выполняем критический анализ
        critic_result = await critic.evaluate_quality(
            risk_type=RiskType.SECURITY,
            original_evaluation=test_evaluation,
            agent_data=agent_data,
            quality_threshold=5.0
        )
        
        print(f"✅ Критик завершил анализ")
        print(f"   Оценка качества: {critic_result.get('quality_score', 'N/A')}/10")
        print(f"   Приемлемо: {critic_result.get('is_acceptable', 'N/A')}")
        print(f"   Найдено проблем: {len(critic_result.get('issues_found', []))}")
        print(f"   Предложений: {len(critic_result.get('improvement_suggestions', []))}")
        
        return critic_result
        
    except Exception as e:
        print(f"❌ Ошибка тестирования критика: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_confidence_manipulation():
    """Тестирование манипуляции уровня уверенности для запуска критика"""
    print("\n⚙️ Тестирование манипуляции уверенности для запуска критика...")
    
    try:
        # Импортируем патч для манипуляции confidence_level
        from src.utils.risk_validation_patch import apply_confidence_and_factors_patch
        
        print("🔧 Применяем патч для снижения уверенности...")
        apply_confidence_and_factors_patch()
        
        # Создаем тестовые данные
        agent_path = create_test_agent_files()
        
        # Запускаем workflow с принудительно низкой уверенностью
        result = await test_full_workflow_with_critic(
            agent_path, 
            quality_threshold=6.0,  # Средний порог
            force_critic=True
        )
        
        return result
        
    except Exception as e:
        print(f"❌ Ошибка тестирования манипуляции уверенности: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_critic_activation_logic():
    """Анализ логики активации критика"""
    print("\n🔍 АНАЛИЗ ЛОГИКИ АКТИВАЦИИ КРИТИКА")
    print("=" * 50)
    
    print("Критик активируется при следующих условиях:")
    print("1. average_quality < quality_threshold")
    print("2. quality_threshold по умолчанию = 7.0")
    print("3. Если уверенность агентов >= 75%, то average_quality >= 7.5")
    print("4. Значит, критик НЕ запустится при высокой уверенности")
    
    print("\nСПОСОБЫ ПРИНУДИТЕЛЬНОГО ЗАПУСКА КРИТИКА:")
    print("A) Снизить QUALITY_THRESHOLD в .env до 6.0 или ниже")
    print("B) Использовать патч для принудительного снижения confidence_level")
    print("C) Передать --quality-threshold при запуске")
    print("D) Модифицировать логику в graph_builder.py")
    
    print("\nРЕКОМЕНДАЦИИ ДЛЯ ТЕСТИРОВАНИЯ:")
    print("1. Установить QUALITY_THRESHOLD=5.0 в .env")
    print("2. Использовать --quality-threshold 5.0 при запуске")
    print("3. Создать агента с заведомо проблемными оценками")

def modify_env_for_critic_testing():
    """Временная модификация настроек для тестирования критика"""
    print("\n⚙️ Настройка окружения для тестирования критика...")
    
    env_file = Path(".env")
    backup_file = Path(".env.backup")
    
    # Создаем бэкап .env файла
    if env_file.exists():
        import shutil
        shutil.copy2(env_file, backup_file)
        print(f"✅ Создан бэкап: {backup_file}")
    
    # Модифицируем настройки
    new_settings = {
        "QUALITY_THRESHOLD": "5.0",  # Низкий порог для гарантированного запуска критика
        "MAX_RETRY_COUNT": "2",
        "LLM_TEMPERATURE": "0.2"  # Чуть больше креативности
    }
    
    try:
        # Читаем существующий .env
        env_content = {}
        if env_file.exists():
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_content[key.strip()] = value.strip()
        
        # Обновляем настройки
        env_content.update(new_settings)
        
        # Записываем обновленный .env
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write("# Временные настройки для тестирования критика\n")
            for key, value in env_content.items():
                f.write(f"{key}={value}\n")
        
        print("✅ Настройки обновлены для тестирования критика:")
        for key, value in new_settings.items():
            print(f"   {key}={value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка модификации .env: {e}")
        return False

def restore_env_backup():
    """Восстановление бэкапа .env файла"""
    env_file = Path(".env")
    backup_file = Path(".env.backup")
    
    if backup_file.exists():
        import shutil
        shutil.copy2(backup_file, env_file)
        backup_file.unlink()  # Удаляем бэкап
        print("✅ Настройки .env восстановлены из бэкапа")
        return True
    else:
        print("⚠️ Бэкап .env не найден")
        return False

async def comprehensive_critic_test():
    """Комплексное тестирование критика"""
    print("\n🎯 КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ АГЕНТА-КРИТИКА")
    print("=" * 60)
    
    results = {}
    
    try:
        # 1. Анализируем логику активации
        analyze_critic_activation_logic()
        
        # 2. Тестируем критика напрямую
        print("\n" + "="*30)
        direct_result = await test_critic_agent_directly()
        results['direct_test'] = direct_result
        
        # 3. Модифицируем окружение
        print("\n" + "="*30)
        env_modified = modify_env_for_critic_testing()
        
        if env_modified:
            # 4. Тестируем с манипуляцией уверенности
            print("\n" + "="*30)
            confidence_result = await test_confidence_manipulation()
            results['confidence_test'] = confidence_result
            
            # 5. Тестируем полный workflow
            print("\n" + "="*30)
            agent_path = create_test_agent_files()
            workflow_result = await test_full_workflow_with_critic(
                agent_path,
                quality_threshold=5.0,
                force_critic=True
            )
            results['workflow_test'] = workflow_result
            
            # Восстанавливаем настройки
            restore_env_backup()
        
        return results
        
    except Exception as e:
        print(f"❌ Критическая ошибка в комплексном тестировании: {e}")
        import traceback
        traceback.print_exc()
        
        # Всегда пытаемся восстановить настройки
        restore_env_backup()
        
        return {"error": str(e)}

def generate_critic_test_report(results: Dict[str, Any]):
    """Генерация отчета о тестировании критика"""
    print("\n📊 ОТЧЕТ О ТЕСТИРОВАНИИ АГЕНТА-КРИТИКА")
    print("=" * 60)
    
    # Анализируем результаты прямого тестирования
    direct_test = results.get('direct_test')
    if direct_test:
        print("✅ ПРЯМОЕ ТЕСТИРОВАНИЕ КРИТИКА:")
        print(f"   Оценка качества: {direct_test.get('quality_score', 'N/A')}/10")
        print(f"   Приемлемость: {direct_test.get('is_acceptable', 'N/A')}")
        
        issues = direct_test.get('issues_found', [])
        print(f"   Найдено проблем: {len(issues)}")
        for i, issue in enumerate(issues[:3], 1):  # Показываем первые 3
            print(f"     {i}. {issue}")
        
        suggestions = direct_test.get('improvement_suggestions', [])
        print(f"   Предложений: {len(suggestions)}")
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"     {i}. {suggestion}")
    
    # Анализируем результаты workflow
    workflow_test = results.get('workflow_test')
    if workflow_test and not workflow_test.get('error'):
        print("\n✅ ТЕСТИРОВАНИЕ ЧЕРЕЗ WORKFLOW:")
        print(f"   Успех: {workflow_test.get('success', False)}")
        
        final_assessment = workflow_test.get('final_assessment', {})
        if final_assessment:
            quality_passed = final_assessment.get('quality_checks_passed', False)
            print(f"   Проверки качества пройдены: {quality_passed}")
            
            processing_time = workflow_test.get('processing_time', 0)
            print(f"   Время обработки: {processing_time:.1f}с")
    
    # Рекомендации
    print("\n💡 РЕКОМЕНДАЦИИ ДЛЯ ПРОДАКШЕНА:")
    
    critic_worked = bool(direct_test and direct_test.get('quality_score') is not None)
    
    if critic_worked:
        print("✅ Критик работает корректно")
        print("📋 Для регулярного запуска критика в продакшене:")
        print("   1. Установите QUALITY_THRESHOLD=6.0 в .env")
        print("   2. Или используйте --quality-threshold 6.0 при запуске")
        print("   3. Мониторьте логи на предмет срабатывания критика")
    else:
        print("❌ Критик не работает как ожидается")
        print("🔧 Рекомендуемые исправления:")
        print("   1. Проверьте подключение к LLM")
        print("   2. Убедитесь в корректности промптов критика")
        print("   3. Проверьте логику активации в graph_builder.py")
    
    print("\n📈 НАСТРОЙКИ ДЛЯ ТЕСТИРОВАНИЯ НА РЕАЛЬНЫХ ДАННЫХ:")
    print("   python main.py assess /path/to/agent --quality-threshold 6.0")
    print("   python main.py assess /path/to/agent --quality-threshold 5.0")
    print("   python main.py assess /path/to/agent --quality-threshold 4.0")

async def main():
    """Основная функция тестирования критика"""
    print("🚀 ТЕСТИРОВАНИЕ АГЕНТА-КРИТИКА")
    print("=" * 50)
    print("Цель: Убедиться что критик запускается и работает корректно")
    print("Проблема: Уверенность агентов не падает ниже 75%, критик не срабатывает")
    print("Решение: Понизить порог качества и протестировать принудительный запуск")
    
    try:
        # Запускаем комплексное тестирование
        results = await comprehensive_critic_test()
        
        # Генерируем отчет
        generate_critic_test_report(results)
        
        print("\n🏁 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        
        # Проверяем успешность
        direct_success = results.get('direct_test') is not None
        workflow_success = (results.get('workflow_test', {}).get('success', False) and 
                          not results.get('workflow_test', {}).get('error'))
        
        if direct_success and workflow_success:
            print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
            return True
        elif direct_success:
            print("⚠️ Критик работает, но есть проблемы с workflow")
            return True
        else:
            print("❌ КРИТИК НЕ РАБОТАЕТ КОРРЕКТНО")
            return False
            
    except Exception as e:
        print(f"💥 КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())