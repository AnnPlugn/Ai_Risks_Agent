# test_main_profiler.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""
CLI интерфейс для тестирования профилирования ИИ-агентов
ИСПРАВЛЕНИЯ:
- Правильные импорты
- Работающий asyncio интерфейс
- Лучшая обработка ошибок
- Детальный вывод результатов
"""

# Загружаем .env файл в самом начале
from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.tree import Tree
from rich.syntax import Syntax

# Наши импорты
from src.workflow.graph_builder_profiler import create_workflow_from_env, test_workflow_health
from src.utils.logger import setup_logging, get_logger
from src.utils.llm_config_manager import get_llm_config_manager, print_env_diagnosis

console = Console()


@click.group()
@click.option('--log-level', default='INFO', help='Уровень логирования (DEBUG/INFO/WARNING/ERROR)')
@click.option('--log-file', default='logs/profiler_test.log', help='Файл логов')
@click.option('--verbose', '-v', is_flag=True, help='Подробный вывод')
@click.pass_context
def cli(ctx, log_level, log_file, verbose):
    """🤖 Тестирование профилирования ИИ-агентов"""
    ctx.ensure_object(dict)

    # Настраиваем логирование
    setup_logging(log_level=log_level, log_file=log_file)
    ctx.obj['logger'] = get_logger()
    ctx.obj['verbose'] = verbose

    # Красивый заголовок
    console.print(Panel.fit(
        "[bold blue]🤖 AI Agent Profiler Test Suite[/bold blue]\n"
        "[cyan]Система тестирования профилирования ИИ-агентов[/cyan]\n"
        "[dim]Архитектура: File System Crawler → Parsers → Context-Aware Chunker → LLM Orchestrator → Output Generator[/dim]",
        title="AI Risk Assessment Profiler",
        border_style="blue"
    ))

    if verbose:
        console.print(f"[dim]📁 Log Level: {log_level}[/dim]")
        console.print(f"[dim]📄 Log File: {log_file}[/dim]")


@cli.command()
@click.pass_context
async def status(ctx):
    """🔍 Проверка статуса системы"""
    verbose = ctx.obj.get('verbose', False)

    console.print("[yellow]🔍 Проверка состояния системы...[/yellow]")

    try:
        # Проверяем конфигурацию
        console.print("\n[bold]📋 Конфигурация LLM:[/bold]")
        manager = get_llm_config_manager()
        config_info = manager.get_info()

        config_table = Table(show_header=True, header_style="bold magenta")
        config_table.add_column("Параметр", style="cyan")
        config_table.add_column("Значение", style="white")

        for key, value in config_info.items():
            config_table.add_row(key, str(value))

        console.print(config_table)

        # Проверяем переменные окружения если verbose
        if verbose:
            console.print("\n[bold]🔧 Диагностика окружения:[/bold]")
            print_env_diagnosis()

        # Проверяем workflow
        console.print("\n[yellow]⚙️ Проверка workflow...[/yellow]")

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("Тестирование workflow...", total=None)

            workflow_healthy = await test_workflow_health()
            progress.update(task, completed=True)

        if workflow_healthy:
            console.print("[green]✅ Workflow работоспособен[/green]")
        else:
            console.print("[red]❌ Workflow не работоспособен[/red]")
            return

        # Проверяем создание workflow
        console.print("\n[yellow]🏗️ Создание workflow...[/yellow]")

        try:
            workflow = create_workflow_from_env()
            workflow_info = workflow.get_workflow_info()

            console.print("[green]✅ Workflow создан успешно[/green]")

            if verbose:
                console.print("\n[bold]📊 Информация о workflow:[/bold]")
                for key, value in workflow_info.items():
                    console.print(f"  [cyan]{key}:[/cyan] {value}")

        except Exception as e:
            console.print(f"[red]❌ Ошибка создания workflow: {e}[/red]")
            if verbose:
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return

        console.print("\n[bold green]🎉 Система готова к работе![/bold green]")

    except Exception as e:
        console.print(f"[red]❌ Ошибка проверки статуса: {e}[/red]")
        if verbose:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")


@cli.command()
@click.argument('source_files', nargs=-1, required=True)
@click.option('--agent-name', '-n', help='Имя анализируемого агента')
@click.option('--output', '-o', help='Файл для сохранения результата (JSON)')
@click.option('--show-files', is_flag=True, help='Показать список обрабатываемых файлов')
@click.option('--save-outputs', is_flag=True, help='Сохранить выходные файлы (отчеты, графы)')
@click.pass_context
async def assess(ctx, source_files, agent_name, output, show_files, save_outputs):
    """🚀 Запуск профилирования ИИ-агента"""
    logger = ctx.obj['logger']
    verbose = ctx.obj.get('verbose', False)

    console.print(f"[bold]🚀 Запуск профилирования агента[/bold]")
    if agent_name:
        console.print(f"[cyan]📛 Имя агента: {agent_name}[/cyan]")

    # 1. Валидация и сбор файлов
    console.print("\n[yellow]📁 Сбор файлов для анализа...[/yellow]")
    validated_files = []

    for file_path in source_files:
        path = Path(file_path)
        if path.exists():
            if path.is_dir():
                # Рекурсивно собираем поддерживаемые файлы
                extensions = ['*.py', '*.js', '*.java', '*.txt', '*.md', '*.json', '*.yaml',
                            '*.yml', '*.docx', '*.pdf', '*.xlsx']
                for ext in extensions:
                    found_files = list(path.rglob(ext))
                    validated_files.extend([str(f.absolute()) for f in found_files])
            else:
                validated_files.append(str(path.absolute()))
        else:
            console.print(f"[red]❌ Не найден: {file_path}[/red]")

    if not validated_files:
        console.print("[red]❌ Не найдено валидных файлов для анализа[/red]")
        return

    # Убираем дубликаты и сортируем
    validated_files = sorted(list(set(validated_files)))

    console.print(f"[green]✅ Найдено файлов: {len(validated_files)}[/green]")

    # Показываем файлы если запрошено
    if show_files or verbose:
        file_tree = Tree("📂 Файлы для анализа")

        # Группируем по типам
        file_types = {}
        for file_path in validated_files:
            ext = Path(file_path).suffix.lower()
            if ext not in file_types:
                file_types[ext] = []
            file_types[ext].append(file_path)

        for ext, files in file_types.items():
            type_branch = file_tree.add(f"📄 {ext} ({len(files)} файлов)")
            for file_path in files[:5]:  # Показываем первые 5
                type_branch.add(Path(file_path).name)
            if len(files) > 5:
                type_branch.add(f"... и еще {len(files) - 5} файлов")

        console.print(file_tree)

    # 2. Создание и проверка workflow
    console.print("\n[yellow]⚙️ Инициализация системы...[/yellow]")

    try:
        workflow = create_workflow_from_env()

        # Показываем конфигурацию LLM
        manager = get_llm_config_manager()
        provider_info = manager.get_info()

        console.print(f"[blue]🤖 Провайдер: {provider_info['provider']}[/blue]")
        console.print(f"[blue]🌐 URL: {provider_info['base_url']}[/blue]")
        console.print(f"[blue]📦 Модель: {provider_info['model']}[/blue]")

        # Проверяем доступность LLM
        print("🔍 Проверка доступности LLM...")
        llm_healthy = await workflow.health_check()

        if not llm_healthy:
            console.print(f"[red]❌ {provider_info['provider']} сервер недоступен[/red]")
            console.print("[yellow]💡 Убедитесь что LLM сервер запущен и доступен[/yellow]")
            return

        console.print(f"[green]✅ {provider_info['provider']} сервер доступен[/green]")

    except Exception as e:
        console.print(f"[red]❌ Ошибка инициализации: {e}[/red]")
        if verbose:
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return

    # 3. Выполнение профилирования
    assessment_id = f"profiler_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    console.print(f"\n[bold]🔄 Выполнение профилирования...[/bold]")
    console.print(f"[dim]Assessment ID: {assessment_id}[/dim]")

    # Progress bar с детальными этапами
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:

        # Основная задача
        main_task = progress.add_task("🔄 Профилирование в процессе...", total=100)

        try:
            # Запускаем профилирование
            progress.update(main_task, completed=10, description="🚀 Запуск workflow...")

            result = await workflow.run_assessment(
                source_files=validated_files,
                agent_name=agent_name,
                assessment_id=assessment_id
            )

            progress.update(main_task, completed=100, description="✅ Профилирование завершено")

        except Exception as e:
            progress.update(main_task, completed=100, description=f"❌ Ошибка: {str(e)[:50]}...")
            console.print(f"\n[red]❌ Исключение во время профилирования: {e}[/red]")
            if verbose:
                console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return

    # 4. Отображение результатов
    if result["success"]:
        await _display_assessment_result(result, output, save_outputs, verbose)

        logger.bind_context(assessment_id, "cli").info(
            f"✅ Профилирование завершено успешно: {assessment_id}"
        )
    else:
        error_msg = result.get('error', 'Неизвестная ошибка')
        console.print(f"\n[red]❌ Профилирование не удалось: {error_msg}[/red]")

        if verbose and 'exception_type' in result:
            console.print(f"[dim]Тип исключения: {result['exception_type']}[/dim]")

        logger.bind_context(assessment_id, "cli").error(
            f"❌ Профилирование не удалось: {error_msg}"
        )


@cli.command()
@click.pass_context
async def demo(ctx):
    """🎯 Демонстрация профилирования на тестовых данных"""
    verbose = ctx.obj.get('verbose', False)

    console.print("[bold]🎯 Демонстрация профилирования[/bold]")
    console.print("[cyan]Создание тестового агента для демонстрации...[/cyan]")

    # Создаем тестовую директорию
    test_dir = Path("demo_agent")
    test_dir.mkdir(exist_ok=True)

    try:
        # Создаем тестовые файлы
        await _create_demo_files(test_dir)

        console.print(f"[green]✅ Создан демо-агент в: {test_dir}[/green]")

        # Запускаем профилирование демо-агента
        console.print("\n[yellow]🚀 Запуск профилирования демо-агента...[/yellow]")

        # Используем существующую команду assess
        ctx.params = {
            'source_files': [str(test_dir)],
            'agent_name': 'DemoAgent',
            'output': 'demo_results.json',
            'show_files': verbose,
            'save_outputs': True
        }

        await assess.callback(
            ctx,
            source_files=[str(test_dir)],
            agent_name='DemoAgent',
            output='demo_results.json',
            show_files=verbose,
            save_outputs=True
        )

    finally:
        # Очистка (опционально)
        if not verbose:  # Сохраняем файлы если verbose для инспекции
            try:
                import shutil
                shutil.rmtree(test_dir)
                console.print(f"[dim]🧹 Очищена тестовая директория[/dim]")
            except Exception as e:
                console.print(f"[yellow]⚠️ Не удалось очистить {test_dir}: {e}[/yellow]")


async def _create_demo_files(test_dir: Path):
    """Создание файлов для демонстрации"""

    # 1. Основной код агента
    agent_code = '''#!/usr/bin/env python3
"""
Демонстрационный ИИ-агент для банковской системы
Анализирует кредитные риски и предоставляет рекомендации
"""

import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class LoanApplication:
    """Заявка на кредит"""
    applicant_id: str
    amount: float
    purpose: str
    income: float
    credit_score: int
    employment_duration: int  # месяцы


class CreditRiskAgent:
    """
    ИИ-агент для оценки кредитных рисков
    Использует LLM для анализа заявок на кредит
    """
    
    def __init__(self, llm_endpoint: str = "http://localhost:1234"):
        self.llm_endpoint = llm_endpoint
        self.system_prompt = """Ты - эксперт по кредитным рискам в банке.
        
        Твоя задача: анализировать заявки на кредит и давать рекомендации.
        
        Учитывай следующие факторы:
        - Кредитная история (credit_score)
        - Доходы заявителя
        - Стабильность занятости
        - Цель кредита
        
        Guardrails:
        - Не дискриминируй по полу, возрасту, национальности
        - Основывайся только на финансовых показателях
        - Не разглашай персональные данные
        - Всегда объясняй решение"""
        
        self.guardrails = [
            "Запрещено использовать защищенные характеристики (пол, раса, религия)",
            "Решения должны быть обоснованными и прозрачными", 
            "Конфиденциальность данных клиентов обязательна",
            "При сомнениях - передавать на ручную проверку"
        ]
    
    async def evaluate_loan_application(self, application: LoanApplication) -> Dict[str, Any]:
        """Оценка заявки на кредит"""
        
        # Формируем промпт для LLM
        analysis_prompt = f"""
        Проанализируй заявку на кредит:
        
        Сумма кредита: {application.amount:,.0f} руб.
        Цель: {application.purpose}
        Доход: {application.income:,.0f} руб/мес
        Кредитный рейтинг: {application.credit_score}/850
        Стаж работы: {application.employment_duration} мес
        
        Дай рекомендацию: одобрить/отклонить/доп.проверка
        """
        
        # Здесь был бы реальный вызов LLM
        # response = await self.call_llm(analysis_prompt)
        
        # Для демо - простая логика
        risk_score = self._calculate_risk_score(application)
        
        if risk_score < 0.3:
            decision = "одобрить"
            reasoning = "Низкий риск: хороший кредитный рейтинг и стабильный доход"
        elif risk_score < 0.7:
            decision = "доп.проверка"
            reasoning = "Средний риск: требуется дополнительная проверка документов"
        else:
            decision = "отклонить"
            reasoning = "Высокий риск: низкий кредитный рейтинг или недостаточный доход"
        
        return {
            "decision": decision,
            "risk_score": risk_score,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
            "agent_version": "1.0",
            "confidence": 0.85
        }
    
    def _calculate_risk_score(self, app: LoanApplication) -> float:
        """Упрощенный расчет риска"""
        
        # Коэффициент долговой нагрузки
        debt_ratio = (app.amount / 12) / app.income if app.income > 0 else 1.0
        
        # Нормализованный кредитный рейтинг
        credit_normalized = (850 - app.credit_score) / 850
        
        # Риск по стажу
        employment_risk = max(0, (12 - app.employment_duration) / 12)
        
        # Общий риск
        risk = (debt_ratio * 0.4 + credit_normalized * 0.4 + employment_risk * 0.2)
        
        return min(1.0, risk)


# Пример использования
async def main():
    agent = CreditRiskAgent()
    
    # Тестовая заявка
    test_application = LoanApplication(
        applicant_id="TEST_001",
        amount=500000,  # 500k рублей
        purpose="покупка автомобиля",
        income=80000,   # 80k рублей в месяц
        credit_score=720,
        employment_duration=24  # 2 года
    )
    
    result = await agent.evaluate_loan_application(test_application)
    print(f"Решение: {result['decision']}")
    print(f"Обоснование: {result['reasoning']}")


if __name__ == "__main__":
    asyncio.run(main())
'''

    (test_dir / "credit_agent.py").write_text(agent_code, encoding='utf-8')

    # 2. Конфигурационный файл
    config = {
        "agent_name": "CreditRiskAgent",
        "version": "1.0.0",
        "description": "ИИ-агент для оценки кредитных рисков в банке",
        "llm_model": "qwen3-4b",
        "max_loan_amount": 5000000,
        "risk_thresholds": {
            "low": 0.3,
            "medium": 0.7,
            "high": 1.0
        },
        "supported_purposes": [
            "покупка автомобиля",
            "покупка недвижимости",
            "рефинансирование",
            "потребительские нужды"
        ]
    }

    (test_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    # 3. Документация
    readme = '''# Credit Risk Agent

## Описание
ИИ-агент для автоматизированной оценки кредитных рисков в банковской системе.

## Возможности
- Анализ заявок на кредит
- Оценка кредитоспособности
- Автоматические рекомендации по одобрению/отклонению
- Соблюдение регуляторных требований

## Архитектура
Агент использует LLM для анализа финансовых данных и принятия решений о кредитных рисках.

### Входные данные
- Сумма кредита
- Доходы заявителя
- Кредитная история
- Цель кредита
- Стаж работы

### Выходные данные  
- Решение (одобрить/отклонить/доп.проверка)
- Оценка риска (0-1)
- Обоснование решения

## Ограничения безопасности
- Не использует защищенные характеристики
- Обеспечивает прозрачность решений
- Соблюдает конфиденциальность данных

## Использование
```python
agent = CreditRiskAgent()
result = await agent.evaluate_loan_application(application)
```

## Техническая спецификация
- **Язык**: Python 3.8+
- **LLM**: qwen3-4b
- **Автономность**: Полуавтономный (требует ручной проверки сложных случаев)
- **Данные**: Внутренние банковские данные, конфиденциальные
'''

    (test_dir / "README.md").write_text(readme, encoding='utf-8')

    # 4. Файл с промптами
    prompts = '''# Системные промпты для CreditRiskAgent

## Основной системный промпт
Ты - эксперт по кредитным рискам в банке.

Твоя задача: анализировать заявки на кредит и давать рекомендации.

Учитывай следующие факторы:
- Кредитная история (credit_score)
- Доходы заявителя  
- Стабильность занятости
- Цель кредита

## Guardrails (Ограничения)
1. Не дискриминируй по полу, возрасту, национальности
2. Основывайся только на финансовых показателях
3. Не разглашай персональные данные
4. Всегда объясняй решение

## Промпт для анализа риска
Проанализируй заявку на кредит и дай рекомендацию.

Учти:
- Коэффициент долговой нагрузки не должен превышать 50%
- Кредитный рейтинг ниже 600 - высокий риск
- Стаж работы менее 6 месяцев - дополнительная проверка

Формат ответа:
- Решение: [одобрить/отклонить/доп.проверка]
- Риск: [низкий/средний/высокий]
- Обоснование: [детальное объяснение]
'''

    (test_dir / "prompts.txt").write_text(prompts, encoding='utf-8')


async def _display_assessment_result(result: Dict[str, Any], output_file: Optional[str] = None,
                                   save_outputs: bool = False, verbose: bool = False):
    """Отображение результатов профилирования"""

    if not result.get("success"):
        console.print("[red]❌ Профилирование не было успешным[/red]")
        return

    final_assessment = result.get("final_assessment", {})
    if not final_assessment:
        console.print("[red]❌ Нет данных для отображения[/red]")
        return

    assessment_id = final_assessment.get('assessment_id', 'unknown')
    agent_profile = final_assessment.get('agent_profile', {})
    processing_time = final_assessment.get('processing_time_seconds', 0)
    profiling_details = final_assessment.get('profiling_details', {})

    # Основная информация
    console.print(Panel(
        f"[bold green]🎯 Профилирование завершено успешно![/bold green]\n\n"
        f"[cyan]Assessment ID:[/cyan] {assessment_id}\n"
        f"[cyan]Время обработки:[/cyan] {processing_time:.2f} секунд\n"
        f"[cyan]Статус:[/cyan] {final_assessment.get('status', 'unknown')}",
        title="📊 Результат профилирования",
        border_style="green"
    ))

    # Профиль агента
    if agent_profile:
        console.print("\n[bold]🤖 Профиль агента:[/bold]")

        profile_table = Table(show_header=True, header_style="bold magenta")
        profile_table.add_column("Параметр", style="cyan", width=25)
        profile_table.add_column("Значение", style="white")

        # Основные поля
        key_fields = [
            ("name", "Имя агента"),
            ("agent_type", "Тип агента"),
            ("description", "Описание"),
            ("llm_model", "LLM модель"),
            ("autonomy_level", "Уровень автономности"),
            ("target_audience", "Целевая аудитория")
        ]

        for field, label in key_fields:
            value = agent_profile.get(field, "Не указано")
            if isinstance(value, list):
                value = ", ".join(map(str, value))
            profile_table.add_row(label, str(value))

        console.print(profile_table)

        # Дополнительная информация если verbose
        if verbose:
            console.print("\n[bold]📋 Дополнительные детали:[/bold]")

            # System prompts
            system_prompts = agent_profile.get("system_prompts", [])
            if system_prompts:
                console.print(f"\n[cyan]📝 Системные промпты ({len(system_prompts)}):[/cyan]")
                for i, prompt in enumerate(system_prompts[:3], 1):
                    preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
                    console.print(f"  {i}. {preview}")
                if len(system_prompts) > 3:
                    console.print(f"  ... и еще {len(system_prompts) - 3} промптов")

            # Guardrails
            guardrails = agent_profile.get("guardrails", [])
            if guardrails:
                console.print(f"\n[cyan]🛡️ Ограничения безопасности ({len(guardrails)}):[/cyan]")
                for i, guardrail in enumerate(guardrails[:3], 1):
                    preview = guardrail[:100] + "..." if len(guardrail) > 100 else guardrail
                    console.print(f"  {i}. {preview}")
                if len(guardrails) > 3:
                    console.print(f"  ... и еще {len(guardrails) - 3} ограничений")

            # External APIs
            external_apis = agent_profile.get("external_apis", [])
            if external_apis:
                console.print(f"\n[cyan]🔌 Внешние API ({len(external_apis)}):[/cyan]")
                for api in external_apis[:5]:
                    console.print(f"  • {api}")
                if len(external_apis) > 5:
                    console.print(f"  ... и еще {len(external_apis) - 5} API")

    # Детали профилирования
    if profiling_details and verbose:
        console.print("\n[bold]🔍 Детали профилирования:[/bold]")

        performance_metrics = profiling_details.get("performance_metrics", {})
        if performance_metrics:
            perf_table = Table(show_header=True, header_style="bold blue")
            perf_table.add_column("Метрика", style="cyan")
            perf_table.add_column("Значение", style="white")

            for key, value in performance_metrics.items():
                if isinstance(value, float):
                    value = f"{value:.2f}"
                perf_table.add_row(key.replace('_', ' ').title(), str(value))

            console.print(perf_table)

    # Сохранение результатов
    if output_file:
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_assessment, f, ensure_ascii=False, indent=2, default=str)

            console.print(f"\n[green]💾 Результаты сохранены в: {output_file}[/green]")

        except Exception as e:
            console.print(f"\n[red]❌ Ошибка сохранения: {e}[/red]")

    # Сохранение выходных файлов (отчеты, графы)
    if save_outputs:
        console.print(f"\n[yellow]📁 Поиск дополнительных файлов результатов...[/yellow]")

        output_dir = Path(f"outputs/{assessment_id}")
        if output_dir.exists():
            output_files = list(output_dir.glob("*"))
            if output_files:
                console.print(f"[green]✅ Найдено {len(output_files)} файлов результатов:[/green]")
                for file_path in output_files:
                    file_size = file_path.stat().st_size
                    console.print(f"  📄 {file_path.name} ({file_size:,} байт)")
            else:
                console.print("[yellow]⚠️ Дополнительные файлы не найдены[/yellow]")
        else:
            console.print("[yellow]⚠️ Директория результатов не создана[/yellow]")


def make_async(f):
    """Декоратор для превращения async функций в sync для Click"""
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


def main():
    """Главная функция CLI"""
    try:
        # Создаем директории
        Path("logs").mkdir(exist_ok=True)
        Path("outputs").mkdir(exist_ok=True)

        # Применяем async декоратор к командам
        assess.callback = make_async(assess.callback)
        status.callback = make_async(status.callback)
        demo.callback = make_async(demo.callback)

        # Запускаем CLI
        cli()

    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Работа прервана пользователем[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Критическая ошибка: {e}[/red]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()