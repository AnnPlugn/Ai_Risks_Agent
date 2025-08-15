"""
CLI интерфейс для тестирования профилирования ИИ-агентов
"""
from dotenv import load_dotenv

load_dotenv()
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
from src.utils.llm_config_manager import get_llm_config_manager
import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.workflow.graph_builder_profiler import create_workflow_from_env
from src.utils.logger import setup_logging, get_logger

console = Console()


@click.group()
@click.option('--log-level', default='INFO', help='Уровень логирования')
@click.option('--log-file', default='logs/profiler_test.log', help='Файл логов')
@click.pass_context
def cli(ctx, log_level, log_file):
    """🤖 Тестирование профилирования ИИ-агентов"""
    ctx.ensure_object(dict)
    setup_logging(log_level=log_level, log_file=log_file)
    ctx.obj['logger'] = get_logger()

    console.print(Panel.fit(
        "[bold blue]🤖 Тест профилирования ИИ-агентов[/bold blue]\n"
        "Упрощённая версия для тестирования профилировщика",
        title="AI Profiler Test",
        border_style="blue"
    ))


@cli.command()
@click.argument('source_files', nargs=-1, required=True)
@click.option('--agent-name', '-n', help='Имя анализируемого агента')
@click.option('--output', '-o', help='Файл для сохранения результата (JSON)')
@click.pass_context
async def assess(ctx, source_files, agent_name, output):
    """Запуск тестирования профилирования"""
    logger = ctx.obj['logger']

    validated_files = []
    for file_path in source_files:
        path = Path(file_path)
        if path.exists():
            if path.is_dir():
                for ext in ['*.py', '*.js', '*.java', '*.txt', '*.md', '*.json', '*.yaml']:
                    validated_files.extend([str(f) for f in path.rglob(ext)])
            else:
                validated_files.append(str(path.absolute()))
        else:
            console.print(f"[red]❌ Файл/папка не найдена: {file_path}[/red]")
            return

    if not validated_files:
        console.print("[red]❌ Не найдено валидных файлов для анализа[/red]")
        return

    console.print(f"[green]📁 Найдено файлов для анализа: {len(validated_files)}[/green]")
    for file_path in validated_files[:10]:
        console.print(f"  • {file_path}")
    if len(validated_files) > 10:
        console.print(f"  ... и еще {len(validated_files) - 10} файлов")

    try:
        console.print("\n[yellow]⚙️ Инициализация workflow...[/yellow]")
        workflow = create_workflow_from_env()

        manager = get_llm_config_manager()
        provider_info = manager.get_info()
        console.print(f"[blue]🤖 Провайдер: {provider_info['provider']}[/blue]")
        console.print(f"[blue]🌐 URL: {provider_info['base_url']}[/blue]")
        console.print(f"[blue]📦 Модель: {provider_info['model']}[/blue]")

        llm_healthy = await workflow.profiler.health_check()
        if not llm_healthy:
            console.print(f"[red]❌ {provider_info['provider']} сервер недоступен[/red]")
            return

        console.print(f"[green]✅ {provider_info['provider']} сервер доступен[/green]")

        assessment_id = f"profiler_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
            task = progress.add_task("🔄 Выполнение профилирования...", total=None)

            result = await workflow.run_assessment(
                source_files=validated_files,
                agent_name=agent_name,
                assessment_id=assessment_id
            )

            progress.update(task, completed=True)

            if result["success"]:
                await _display_assessment_result(result, output)
                logger.bind_context(assessment_id, "cli").info(
                    f"✅ Профилирование завершено: {assessment_id}"
                )
            else:
                console.print(f"[red]❌ Ошибка профилирования: {result.get('error', 'Неизвестная ошибка')}[/red]")

    except Exception as e:
        console.print(f"\n[red]❌ Неожиданная ошибка: {e}[/red]")
        logger.bind_context(assessment_id, "cli").error(f"Ошибка CLI: {e}")


async def _display_assessment_result(result, output_file=None):
    """Отображение результатов профилирования"""
    assessment = result.get("final_assessment", {})
    if not assessment:
        console.print("[red]❌ Нет данных для отображения[/red]")
        return

    assessment_id = assessment.get('assessment_id', 'unknown')
    profile = assessment.get('agent_profile', {})
    processing_time = assessment.get('processing_time_seconds', 0)

    console.print(Panel(
        f"[bold green]🎯 Профилирование завершено![/bold green]\n\n"
        f"Assessment ID: {assessment_id}\n"
        f"Имя агента: {profile.get('name', 'Unknown')}\n"
        f"Тип агента: {profile.get('agent_type', 'Unknown')}\n"
        f"Время обработки: {processing_time:.1f} секунд",
        title="📊 Результат профилирования",
        border_style="green"
    ))

    if profile:
        table = Table(title="🔍 Детали профиля")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="white")
        for key, value in profile.items():
            table.add_row(key, str(value))
        console.print(table)

    if output_file:
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(assessment, f, ensure_ascii=False, indent=2, default=str)
            console.print(f"\n[green]💾 Результаты сохранены в {output_file}[/green]")
        except Exception as e:
            console.print(f"\n[red]❌ Ошибка сохранения: {e}[/red]")


def main():
    """Главная функция CLI"""
    try:
        Path("logs").mkdir(exist_ok=True)
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 До свидания![/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Критическая ошибка: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    def make_async(f):
        def wrapper(*args, **kwargs):
            return asyncio.run(f(*args, **kwargs))

        return wrapper


    assess.callback = make_async(assess.callback)
    main()