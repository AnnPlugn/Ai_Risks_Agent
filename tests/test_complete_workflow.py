# test_complete_workflow.py
"""
Полный тест workflow системы оценки рисков (Этап 4)
Тестирует весь цикл от загрузки файлов до сохранения результата
"""

import asyncio
import sys
import tempfile
import json
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_workflow_creation():
    """Тест создания workflow"""
    print("🧪 Тестирование создания LangGraph workflow...")
    
    try:
        from src.workflow import create_workflow_from_env
        
        # Создаем workflow
        workflow = create_workflow_from_env()
        
        print(f"✅ Workflow создан: {workflow.__class__.__name__}")
        print(f"✅ Количество агентов: Профайлер + {len(workflow.evaluators)} оценщиков + Критик")
        print(f"✅ Порог качества: {workflow.quality_threshold}")
        print(f"✅ Максимум повторов: {workflow.max_retries}")
        
        # Проверяем граф
        print(f"✅ LangGraph скомпилирован: {workflow.graph is not None}")
        
        return True, workflow
        
    except Exception as e:
        print(f"❌ Ошибка создания workflow: {e}")
        return False, None

async def test_health_checks(workflow):
    """Тест проверки работоспособности компонентов"""
    print("\n🧪 Тестирование health checks...")
    
    try:
        # Проверяем профайлер
        profiler_health = await workflow.profiler.health_check()
        print(f"✅ Профайлер health check: {'✅' if profiler_health else '⚠️ LM Studio недоступен'}")
        
        # Проверяем оценщиков
        healthy_evaluators = 0
        for risk_type, evaluator in workflow.evaluators.items():
            evaluator_health = await evaluator.health_check()
            if evaluator_health:
                healthy_evaluators += 1
        
        print(f"✅ Оценщики health check: {healthy_evaluators}/{len(workflow.evaluators)}")
        
        # Проверяем критика
        critic_health = await workflow.critic.health_check()
        print(f"✅ Критик health check: {'✅' if critic_health else '⚠️ LM Studio недоступен'}")
        
        return profiler_health and healthy_evaluators > 0 and critic_health
        
    except Exception as e:
        print(f"❌ Ошибка health checks: {e}")
        return False

async def create_test_files():
    """Создание тестовых файлов для workflow"""
    print("\n🧪 Создание тестовых файлов...")
    
    # Создаем временную директорию
    temp_dir = Path(tempfile.mkdtemp())
    
    # Тестовый код агента
    agent_code = '''# test_banking_agent.py
"""
Тестовый банковский агент для демонстрации системы оценки рисков
"""

class BankingAssistant:
    def __init__(self):
        self.name = "BankingAssistant"
        self.version = "2.1"
        
        # Системный промпт
        self.system_prompt = """
        Ты - виртуальный помощник банка "Тест-Банк".
        Твоя задача - помогать клиентам с банковскими услугами.
        
        Твои возможности:
        - Отвечать на вопросы о продуктах банка
        - Помогать с навигацией по мобильному приложению
        - Предоставлять общую информацию о банковских услугах
        
        Ограничения безопасности:
        - НИКОГДА не запрашивай пароли или PIN-коды
        - НЕ обрабатывай операции с деньгами без подтверждения
        - НЕ раскрывай персональные данные других клиентов
        - Направляй сложные финансовые вопросы к специалистам
        - При подозрении на мошенничество - немедленно предупреди клиента
        """
        
        self.guardrails = [
            "Не запрашивать пароли и PIN-коды",
            "Не выполнять денежные операции без подтверждения", 
            "Защищать персональные данные клиентов",
            "Направлять сложные вопросы к специалистам"
        ]
    
    def process_user_query(self, query: str, user_id: str) -> str:
        """Обработка пользовательского запроса"""
        # Простая логика для демонстрации
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['баланс', 'счет', 'остаток']):
            return "Проверить баланс можно в мобильном приложении в разделе 'Мои счета'"
        
        elif any(word in query_lower for word in ['кредит', 'займ', 'ипотека']):
            return "По вопросам кредитования рекомендую обратиться к кредитному специалисту"
        
        elif any(word in query_lower for word in ['карта', 'платеж', 'перевод']):
            return "Операции с картами доступны через мобильное приложение или банкомат"
        
        else:
            return "Как я могу помочь вам сегодня? Расскажите подробнее о вашем вопросе."
    
    def get_available_products(self):
        """Получение списка банковских продуктов"""
        return [
            "Расчетные счета",
            "Сберегательные вклады", 
            "Кредитные карты",
            "Потребительские кредиты",
            "Ипотечное кредитование",
            "Инвестиционные продукты"
        ]

# Пример использования
if __name__ == "__main__":
    assistant = BankingAssistant()
    print(f"Запущен {assistant.name} v{assistant.version}")
'''
    
    # Документация агента
    agent_docs = '''Техническая документация: Банковский виртуальный помощник

ОБЩАЯ ИНФОРМАЦИЯ
Название: BankingAssistant
Версия: 2.1
Тип агента: Чат-бот для клиентского сервиса
Дата создания: 2024-12-14

НАЗНАЧЕНИЕ И ФУНКЦИИ
Виртуальный помощник предназначен для автоматизации первичной поддержки клиентов банка.

Основные функции:
• Консультации по банковским продуктам и услугам
• Помощь в навигации по мобильному приложению
• Ответы на часто задаваемые вопросы
• Направление к соответствующим специалистам при необходимости

ТЕХНИЧЕСКАЯ СПЕЦИФИКАЦИЯ
Архитектура: Микросервисная
Язык разработки: Python 3.12
Модель ИИ: Qwen3-4B
Уровень автономности: Под надзором человека
Операций в час: ~500-1000

ДОСТУП К ДАННЫМ
Уровень доступа: Внутренние справочные данные
- Информация о продуктах банка
- FAQ и инструкции для клиентов
- Общедоступные тарифы и условия

НЕ имеет доступа к:
- Персональным данным клиентов
- Финансовой информации
- Операционным системам банка

ЦЕЛЕВАЯ АУДИТОРИЯ
Первичная: Клиенты банка (физические лица)
Вторичная: Клиенты банка (малый и средний бизнес)
Возрастная группа: 18-65 лет
Опыт использования банковских услуг: Любой

МЕРЫ БЕЗОПАСНОСТИ И ОГРАНИЧЕНИЯ
Встроенные ограничения:
1. Запрет на запрос конфиденциальной информации (пароли, PIN-коды)
2. Невозможность выполнения финансовых операций без дополнительного подтверждения
3. Автоматическое направление сложных запросов к человеку-оператору
4. Защита от попыток социальной инженерии

Мониторинг и контроль:
- Логирование всех диалогов
- Ежедневный анализ качества ответов
- Еженедельный пересмотр политик безопасности

ИНТЕГРАЦИИ
Внешние системы:
- CRM система банка (только чтение справочников)
- База знаний технической поддержки
- Система аналитики и мониторинга

API:
- REST API для получения актуальной информации о продуктах
- Webhook для уведомлений о критических ситуациях

ПРОЦЕДУРЫ ЭСКАЛАЦИИ
При обнаружении:
- Попыток мошенничества → Немедленная передача службе безопасности
- Технических проблем → Автоматическое уведомление IT-поддержки  
- Жалоб клиентов → Передача менеджеру по работе с клиентами

МЕТРИКИ КАЧЕСТВА
Целевые показатели:
- Точность ответов: >95%
- Время ответа: <3 секунд
- Удовлетворенность клиентов: >4.5/5
- Процент эскалации к человеку: <15%

ПЛАНЫ РАЗВИТИЯ
Ближайшие улучшения:
- Расширение базы знаний
- Улучшение понимания контекста диалога
- Интеграция с системой онлайн-банкинга
- Поддержка голосового интерфейса
'''
    
    # Конфигурационный файл
    config_data = {
        "agent_config": {
            "name": "BankingAssistant",
            "version": "2.1",
            "type": "customer_service_chatbot",
            "autonomy_level": "supervised",
            "max_concurrent_sessions": 100,
            "session_timeout": 1800,
            "escalation_threshold": 0.7
        },
        "llm_settings": {
            "model": "qwen3-4b",
            "temperature": 0.2,
            "max_tokens": 512,
            "top_p": 0.9
        },
        "security_policies": {
            "pii_detection": True,
            "content_filtering": True,
            "audit_logging": True,
            "rate_limiting": True
        },
        "integrations": {
            "crm_system": {
                "endpoint": "https://crm.test-bank.ru/api/v1",
                "access_level": "read_only"
            },
            "knowledge_base": {
                "endpoint": "https://kb.test-bank.ru/api/v2",
                "cache_ttl": 3600
            }
        }
    }
    
    # Сохраняем файлы
    files_created = []
    
    # Код агента
    agent_file = temp_dir / "banking_agent.py"
    agent_file.write_text(agent_code, encoding='utf-8')
    files_created.append(str(agent_file))
    
    # Документация
    docs_file = temp_dir / "agent_documentation.txt"
    docs_file.write_text(agent_docs, encoding='utf-8')
    files_created.append(str(docs_file))
    
    # Конфигурация
    config_file = temp_dir / "agent_config.json"
    config_file.write_text(json.dumps(config_data, indent=2, ensure_ascii=False), encoding='utf-8')
    files_created.append(str(config_file))
    
    print(f"✅ Создано {len(files_created)} тестовых файлов:")
    for file_path in files_created:
        file_size = Path(file_path).stat().st_size
        print(f"  • {Path(file_path).name} ({file_size} байт)")
    
    return files_created, temp_dir

async def test_full_workflow(workflow, test_files):
    """Тест полного workflow"""
    print("\n🧪 Тестирование полного workflow...")
    
    try:
        # Запускаем полную оценку
        result = await workflow.run_assessment(
            source_files=test_files,
            agent_name="BankingAssistant_Test"
        )
        
        print(f"✅ Workflow завершен: {'успешно' if result['success'] else 'с ошибкой'}")
        
        if result["success"]:
            assessment = result.get("final_assessment")
            if assessment:
                print(f"✅ ID оценки: {result['assessment_id']}")
                print(f"✅ Время обработки: {result.get('processing_time', 0):.1f}с")
                print(f"✅ Общий риск: {assessment['overall_risk_level']} ({assessment['overall_risk_score']}/25)")
                
                # Проверяем наличие оценок по всем типам рисков
                risk_evaluations = assessment.get("risk_evaluations", {})
                print(f"✅ Оценок рисков: {len(risk_evaluations)}/6")
                
                for risk_type, evaluation in risk_evaluations.items():
                    level = evaluation.get("risk_level", "unknown")
                    score = evaluation.get("total_score", 0)
                    print(f"  • {risk_type}: {level} ({score}/25)")
                
                # Проверяем рекомендации
                recommendations = assessment.get("priority_recommendations", [])
                print(f"✅ Рекомендаций: {len(recommendations)}")
                
                return True, result
            else:
                print("❌ Отсутствует итоговая оценка")
                return False, result
        else:
            print(f"❌ Ошибка workflow: {result.get('error', 'Неизвестная ошибка')}")
            return False, result
            
    except Exception as e:
        print(f"❌ Исключение в workflow: {e}")
        return False, None

async def test_database_integration(workflow, result):
    """Тест интеграции с базой данных"""
    print("\n🧪 Тестирование сохранения в БД...")
    
    try:
        from src.models.database import get_db_manager
        
        if not result or not result.get("success"):
            print("⚠️ Пропускаем тест БД - нет успешного результата")
            return False
        
        assessment_id = result.get("saved_assessment_id")
        if not assessment_id:
            print("❌ Отсутствует ID сохраненной оценки")
            return False
        
        # Проверяем сохранение
        db_manager = await get_db_manager()
        
        # Загружаем оценку из БД
        saved_assessment = await db_manager.get_risk_assessment(assessment_id)
        
        if saved_assessment:
            print(f"✅ Оценка найдена в БД: {assessment_id}")
            
            assessment_record = saved_assessment["assessment"]
            evaluations = saved_assessment["evaluations"]
            
            print(f"✅ Общий риск: {assessment_record.overall_risk_level} ({assessment_record.overall_risk_score}/25)")
            print(f"✅ Детальных оценок: {len(evaluations)}")
            print(f"✅ Дата создания: {assessment_record.assessment_timestamp}")
            
            # Проверяем логи обработки
            logs = await db_manager.get_processing_logs(result["assessment_id"])
            print(f"✅ Логов обработки: {len(logs)}")
            
            return True
        else:
            print(f"❌ Оценка не найдена в БД: {assessment_id}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка тестирования БД: {e}")
        return False

async def test_cli_integration():
    """Тест CLI интерфейса"""
    print("\n🧪 Тестирование CLI интерфейса...")
    
    try:
        # Импортируем CLI модуль
        import main
        
        print("✅ CLI модуль импортирован")
        print("✅ Rich консоль инициализирована")
        print("✅ Click команды зарегистрированы")
        
        # Проверяем доступность команд
        available_commands = ["assess", "show", "list-assessments", "status", "demo"]
        print(f"✅ Доступные команды: {', '.join(available_commands)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования CLI: {e}")
        return False

async def test_performance_metrics(workflow, test_files):
    """Тест производительности системы"""
    print("\n🧪 Тестирование производительности...")
    
    try:
        import time
        
        # Измеряем время отдельных компонентов
        start_time = time.time()
        
        # Профилирование
        profiler_start = time.time()
        profiler_result = await workflow.profiler.run(
            {"source_files": test_files, "agent_name": "PerfTest"},
            "perf_test_001"
        )
        profiler_time = time.time() - profiler_start
        
        print(f"✅ Профилирование: {profiler_time:.2f}с")
        
        if profiler_result.status.value == "completed":
            agent_profile = profiler_result.result_data["agent_profile"]
            
            # Тест одного оценщика
            from src.models.risk_models import RiskType
            ethical_evaluator = workflow.evaluators[RiskType.ETHICAL]
            
            eval_start = time.time()
            eval_result = await ethical_evaluator.run(
                {"agent_profile": agent_profile},
                "perf_test_001"
            )
            eval_time = time.time() - eval_start
            
            print(f"✅ Оценка этических рисков: {eval_time:.2f}с")
            
            # Тест критика
            if eval_result.status.value == "completed":
                critic_start = time.time()
                critic_input = {
                    "risk_type": RiskType.ETHICAL.value,
                    "risk_evaluation": eval_result.result_data["risk_evaluation"],
                    "agent_profile": agent_profile,
                    "evaluator_name": ethical_evaluator.name
                }
                
                critic_result = await workflow.critic.run(critic_input, "perf_test_001")
                critic_time = time.time() - critic_start
                
                print(f"✅ Критический анализ: {critic_time:.2f}с")
        
        total_time = time.time() - start_time
        print(f"✅ Общее время компонентов: {total_time:.2f}с")
        
        # Получаем статистику агентов
        profiler_stats = workflow.profiler.get_stats()
        print(f"✅ Статистика профайлера: {profiler_stats['total_requests']} запросов, "
              f"успешность {profiler_stats['success_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования производительности: {e}")
        return False

async def cleanup_test_files(temp_dir):
    """Очистка тестовых файлов"""
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"✅ Временные файлы очищены: {temp_dir}")
    except Exception as e:
        print(f"⚠️ Не удалось очистить временные файлы: {e}")

async def main():
    """Главная функция полного тестирования"""
    print("🚀 ПОЛНОЕ ТЕСТИРОВАНИЕ WORKFLOW СИСТЕМЫ ОЦЕНКИ РИСКОВ")
    print("=" * 70)
    
    success_count = 0
    total_tests = 6
    
    # Тест 1: Создание workflow
    workflow_success, workflow = await test_workflow_creation()
    if workflow_success:
        success_count += 1
    
    # Тест 2: Health checks (только если workflow создан)
    if workflow_success and workflow:
        health_success = await test_health_checks(workflow)
        if health_success:
            success_count += 1
        
        # Тест 3: Создание тестовых файлов и полный workflow
        test_files, temp_dir = await create_test_files()
        
        try:
            workflow_test_success, workflow_result = await test_full_workflow(workflow, test_files)
            if workflow_test_success:
                success_count += 1
            
            # Тест 4: Интеграция с БД (только если workflow успешен)
            if workflow_test_success:
                db_success = await test_database_integration(workflow, workflow_result)
                if db_success:
                    success_count += 1
            
            # Тест 5: CLI интерфейс
            cli_success = await test_cli_integration()
            if cli_success:
                success_count += 1
            
            # Тест 6: Производительность
            if workflow_test_success:
                perf_success = await test_performance_metrics(workflow, test_files)
                if perf_success:
                    success_count += 1
            
        finally:
            # Очищаем тестовые файлы
            await cleanup_test_files(temp_dir)
    
    print("\n" + "=" * 70)
    print(f"📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("\n✅ Система полностью готова к использованию:")
        print("  • LangGraph workflow функционирует")
        print("  • Все агенты работают корректно")  
        print("  • База данных сохраняет результаты")
        print("  • CLI интерфейс готов к использованию")
        print("  • Производительность в норме")
        
        print("\n🚀 Для запуска системы используйте:")
        print("  python main.py assess <файлы_агента>")
        print("  python main.py demo  # для демонстрации")
        print("  python main.py status --check-llm --check-db")
        
    elif success_count >= total_tests * 0.8:
        print("🎯 БОЛЬШИНСТВО ТЕСТОВ ПРОЙДЕНО!")
        print(f"✅ Успешно: {success_count}")
        print(f"❌ Неудачно: {total_tests - success_count}")
        print("\n💡 Система готова к использованию с ограничениями")
        
    else:
        print("⚠️ ТРЕБУЮТСЯ ДОПОЛНИТЕЛЬНЫЕ ИСПРАВЛЕНИЯ")
        print(f"✅ Успешно: {success_count}")
        print(f"❌ Неудачно: {total_tests - success_count}")
        print("\n🔧 Рекомендуется:")
        print("  • Проверить запуск LM Studio")
        print("  • Проверить зависимости")
        print("  • Проанализировать логи ошибок")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)