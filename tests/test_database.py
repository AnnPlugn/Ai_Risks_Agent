# test_database.py
"""
Простой тест для проверки что данные сохраняются в БД
"""

import asyncio
import sys
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models.database import get_db_manager
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


async def test_database():
    """Простая проверка базы данных"""
    
    console.print(Panel.fit(
        "[bold blue]🗄️ Тест базы данных[/bold blue]",
        border_style="blue"
    ))
    
    try:
        # Подключаемся к БД (get_db_manager уже инициализирует)
        db_manager = await get_db_manager()
        console.print("[green]✅ Подключение к БД успешно[/green]")
        
        # Проверяем таблицы
        async with db_manager.async_session() as session:
            # Проверяем есть ли записи в основных таблицах
            from sqlalchemy import text
            
            tables = ['agent_profiles', 'risk_assessments', 'risk_evaluations', 'critic_evaluations']
            
            stats_table = Table(title="📊 Статистика таблиц")
            stats_table.add_column("Таблица", style="cyan")
            stats_table.add_column("Записей", style="white")
            
            for table in tables:
                try:
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    stats_table.add_row(table, str(count))
                except Exception as e:
                    stats_table.add_row(table, f"Ошибка: {str(e)[:50]}")
            
            console.print(stats_table)
        
        # Получаем последние 3 оценки если есть
        try:
            from sqlalchemy import select, desc
            from src.models.database import RiskAssessmentDB
            
            async with db_manager.async_session() as session:
                stmt = select(RiskAssessmentDB).order_by(desc(RiskAssessmentDB.assessment_timestamp)).limit(3)
                result = await session.execute(stmt)
                assessments = result.scalars().all()
                
                if assessments:
                    console.print(f"\n[green]📋 Найдено {len(assessments)} недавних оценок:[/green]")
                    
                    recent_table = Table()
                    recent_table.add_column("ID", style="cyan")
                    recent_table.add_column("Риск", style="white")
                    recent_table.add_column("Дата", style="green")
                    
                    for assessment in assessments:
                        recent_table.add_row(
                            assessment.id[:8] + "...",
                            f"{assessment.overall_risk_level} ({assessment.overall_risk_score})",
                            str(assessment.assessment_timestamp)[:19]
                        )
                    
                    console.print(recent_table)
                else:
                    console.print("[yellow]📭 Пока нет сохраненных оценок[/yellow]")
                    console.print("[dim]Запустите: python main.py assess <files> чтобы создать первую оценку[/dim]")
        
        except Exception as e:
            console.print(f"[yellow]⚠️ Ошибка получения оценок: {e}[/yellow]")
        
        await db_manager.close()
        console.print("\n[green]✅ Тест базы данных завершен успешно[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Ошибка теста БД: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    asyncio.run(test_database())