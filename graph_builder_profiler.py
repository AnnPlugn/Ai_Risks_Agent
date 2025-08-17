# src/workflow/graph_builder_profiler.py - ИСПРАВЛЕННАЯ ВЕРСИЯ
"""
LangGraph workflow для тестирования профилирования ИИ-агентов
ИСПРАВЛЕНИЯ:
- Правильные импорты
- Работающие узлы
- Корректная обработка состояний
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Добавляем путь к src в Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from ..utils.llm_config_manager import get_llm_config_manager
from ..models.risk_models import WorkflowState, AgentProfile, ProcessingStatus
from ..agents.profiler_agent import create_profiler_from_env, create_profiler_node_function
from ..utils.logger import get_langgraph_logger, log_graph_node


class ProfilerWorkflow:
    """
    ИСПРАВЛЕННЫЙ Workflow для профилирования ИИ-агентов
    Архитектура: File System Crawler → Parsers → Context-Aware Chunker → LLM Orchestrator → Output Generator
    """

    def __init__(self):
        print("🔧 Инициализация ProfilerWorkflow...")

        # Получаем конфигурацию LLM
        self.config_manager = get_llm_config_manager()
        self.llm_base_url = self.config_manager.get_base_url()
        self.llm_model = self.config_manager.get_model()

        print(f"🤖 LLM Provider: {self.config_manager.get_provider().value}")
        print(f"🌐 LLM URL: {self.llm_base_url}")
        print(f"📦 LLM Model: {self.llm_model}")

        # Создаем профайлер
        self.profiler = create_profiler_from_env()
        self.graph_logger = get_langgraph_logger()

        # Строим граф
        self.graph = self._build_graph()
        print("✅ ProfilerWorkflow инициализирован")

    def _build_graph(self) -> CompiledStateGraph:
        """Построение графа профилирования"""
        print("🏗️ Построение LangGraph для профилирования...")

        # Создаем граф состояний
        workflow = StateGraph(WorkflowState)

        # Добавляем узлы
        print("📍 Добавление узлов...")
        self._add_nodes(workflow)

        # Добавляем рёбра
        print("🔗 Добавление рёбер...")
        self._add_edges(workflow)

        # Устанавливаем точки входа и выхода
        workflow.set_entry_point("initialization")
        workflow.set_finish_point("finalization")

        # Компилируем граф
        print("⚙️ Компиляция графа...")
        compiled_graph = workflow.compile()
        print("✅ Граф скомпилирован успешно")

        return compiled_graph

    def _add_nodes(self, workflow: StateGraph):
        """Добавление узлов в граф"""

        # Узел инициализации
        workflow.add_node("initialization", self._initialization_node)
        print("  ✅ initialization")

        # Узел профилирования (главный узел архитектуры)
        profiler_node = create_profiler_node_function(self.profiler)
        workflow.add_node("profiling", profiler_node)
        print("  ✅ profiling")

        # Узел финализации
        workflow.add_node("finalization", self._finalization_node)
        print("  ✅ finalization")

        # Узел обработки ошибок
        workflow.add_node("error_handling", self._error_handling_node)
        print("  ✅ error_handling")

    def _add_edges(self, workflow: StateGraph):
        """Добавление рёбер между узлами"""

        # Прямой путь
        workflow.add_edge("initialization", "profiling")
        print("  ✅ initialization → profiling")

        # Условный переход от профилирования
        workflow.add_conditional_edges(
            "profiling",
            self._routing_condition,
            {
                "finalization": "finalization",
                "error": "error_handling"
            }
        )
        print("  ✅ profiling → [finalization|error]")

        # Завершение
        workflow.add_edge("finalization", END)
        workflow.add_edge("error_handling", END)
        print("  ✅ finalization → END")
        print("  ✅ error_handling → END")

    @log_graph_node("initialization")
    async def _initialization_node(self, state: WorkflowState) -> WorkflowState:
        """Узел инициализации профилирования"""

        # Генерируем assessment_id если не указан
        assessment_id = state.get("assessment_id")
        if not assessment_id:
            assessment_id = f"profiler_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.graph_logger.log_graph_start(assessment_id, "profiler_workflow")

        # Проверяем наличие файлов
        source_files = state.get("source_files", [])
        if not source_files:
            state.update({
                "current_step": "error",
                "error_message": "Не предоставлены файлы для анализа",
                "assessment_id": assessment_id
            })
            return state

        # Валидируем файлы
        validated_files = []
        for file_path in source_files:
            file_path_obj = Path(file_path)
            if file_path_obj.exists():
                validated_files.append(str(file_path_obj.absolute()))
            else:
                print(f"⚠️ Файл не найден: {file_path}")

        if not validated_files:
            state.update({
                "current_step": "error",
                "error_message": "Все указанные файлы недоступны",
                "assessment_id": assessment_id
            })
            return state

        # Обновляем состояние
        state.update({
            "assessment_id": assessment_id,
            "source_files": validated_files,
            "current_step": "profiling",
            "start_time": datetime.now()
        })

        self.graph_logger.log_workflow_step(
            assessment_id,
            "initialization",
            f"Подготовлено {len(validated_files)} файлов"
        )

        return state

    @log_graph_node("finalization")
    async def _finalization_node(self, state: WorkflowState) -> WorkflowState:
        """Узел финализации результатов"""

        assessment_id = state.get("assessment_id", "unknown")
        start_time = state.get("start_time", datetime.now())
        processing_time = (datetime.now() - start_time).total_seconds()

        # Извлекаем результаты профилирования
        agent_profile = state.get("agent_profile", {})
        profiling_result = state.get("profiling_result", {})

        # Создаем итоговую оценку
        final_assessment = {
            "assessment_id": assessment_id,
            "agent_profile": agent_profile,
            "profiling_details": profiling_result,
            "processing_time_seconds": processing_time,
            "timestamp": datetime.now().isoformat(),
            "status": "completed" if agent_profile else "failed",
            "workflow_type": "profiling_only"
        }

        # Обновляем состояние
        state.update({
            "final_assessment": final_assessment,
            "current_step": "completed",
            "processing_time": processing_time
        })

        # Логируем завершение
        self.graph_logger.log_graph_completion(
            assessment_id,
            processing_time,
            1 if agent_profile else 0
        )

        if agent_profile:
            agent_name = agent_profile.get("name", "Unknown")
            agent_type = agent_profile.get("agent_type", "unknown")
            print(f"✅ Профилирование завершено: {agent_name} ({agent_type})")
        else:
            print("❌ Профилирование не удалось")

        return state

    @log_graph_node("error_handling")
    async def _error_handling_node(self, state: WorkflowState) -> WorkflowState:
        """Узел обработки ошибок"""

        assessment_id = state.get("assessment_id", "unknown")
        error_message = state.get("error_message", "Неизвестная ошибка")

        # Создаем результат с ошибкой
        final_assessment = {
            "assessment_id": assessment_id,
            "status": "failed",
            "error_message": error_message,
            "timestamp": datetime.now().isoformat(),
            "workflow_type": "profiling_only"
        }

        state.update({
            "final_assessment": final_assessment,
            "current_step": "failed"
        })

        print(f"❌ Обработка ошибки: {error_message}")
        return state

    def _routing_condition(self, state: WorkflowState) -> str:
        """Условие маршрутизации после профилирования"""

        current_step = state.get("current_step", "unknown")
        error_message = state.get("error_message")

        if error_message or current_step == "error":
            return "error"
        elif current_step == "finalization":
            return "finalization"
        else:
            # По умолчанию считаем что нужна финализация
            return "finalization"

    async def run_assessment(
        self,
        source_files: List[str],
        agent_name: Optional[str] = None,
        assessment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Запуск полного цикла профилирования

        Args:
            source_files: Список файлов/папок для анализа
            agent_name: Имя анализируемого агента (опционально)
            assessment_id: ID оценки (генерируется автоматически если не указан)

        Returns:
            Результат профилирования
        """

        print(f"🚀 Запуск профилирования для {len(source_files)} источников...")

        # Создаем начальное состояние
        initial_state = WorkflowState(
            source_files=source_files,
            preliminary_agent_name=agent_name or "Unknown_Agent",
            assessment_id=assessment_id,
            current_step="initialization"
        )

        try:
            # Проверяем здоровье LLM перед запуском
            print("🔍 Проверка доступности LLM...")
            is_healthy = await self.profiler.health_check()
            if not is_healthy:
                provider = self.config_manager.get_provider().value
                return {
                    "success": False,
                    "error": f"{provider} сервер недоступен",
                    "assessment_id": initial_state.assessment_id
                }

            print(f"✅ LLM сервер доступен")

            # Запускаем граф
            print("⚙️ Выполнение LangGraph workflow...")
            final_state = await self.graph.ainvoke(initial_state.dict())

            # Извлекаем результаты
            final_assessment = final_state.get("final_assessment", {})
            success = final_assessment.get("status") == "completed"

            return {
                "success": success,
                "assessment_id": final_state.get("assessment_id"),
                "final_assessment": final_assessment,
                "processing_time": final_state.get("processing_time", 0),
                "current_step": final_state.get("current_step", "unknown")
            }

        except Exception as e:
            print(f"❌ Исключение в workflow: {e}")
            return {
                "success": False,
                "error": str(e),
                "assessment_id": initial_state.assessment_id,
                "exception_type": type(e).__name__
            }

    async def health_check(self) -> bool:
        """Проверка работоспособности workflow"""
        try:
            # Проверяем профайлер
            profiler_healthy = await self.profiler.health_check()
            if not profiler_healthy:
                return False

            # Проверяем что граф скомпилирован
            if not self.graph:
                return False

            return True

        except Exception:
            return False

    def get_workflow_info(self) -> Dict[str, Any]:
        """Получение информации о workflow"""
        return {
            "workflow_type": "profiling_only",
            "llm_provider": self.config_manager.get_provider().value,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "nodes": ["initialization", "profiling", "finalization", "error_handling"],
            "architecture_components": [
                "File System Crawler",
                "Code Parser",
                "Doc Parser",
                "Context-Aware Chunker",
                "LLM Orchestrator",
                "Output Generator"
            ]
        }


# ===============================
# Фабричные функции
# ===============================

def create_workflow_from_env() -> ProfilerWorkflow:
    """Создание workflow из переменных окружения"""
    return ProfilerWorkflow()


def create_test_workflow() -> ProfilerWorkflow:
    """Создание workflow для тестирования"""
    return ProfilerWorkflow()


# ===============================
# Утилитарные функции для тестирования
# ===============================

async def test_workflow_health():
    """Быстрый тест работоспособности workflow"""
    try:
        workflow = create_workflow_from_env()
        is_healthy = await workflow.health_check()

        if is_healthy:
            print("✅ Workflow работоспособен")
            return True
        else:
            print("❌ Workflow не работоспособен")
            return False

    except Exception as e:
        print(f"❌ Ошибка проверки workflow: {e}")
        return False


async def test_minimal_profiling():
    """Минимальный тест профилирования"""

    # Создаем тестовые файлы
    test_dir = Path("test_agent")
    test_dir.mkdir(exist_ok=True)

    # Простой Python файл
    test_file = test_dir / "agent.py"
    test_file.write_text('''
# Тестовый ИИ-агент
class TestAgent:
    def __init__(self):
        self.system_prompt = "Ты полезный ассистент"
        self.guardrails = ["Не отвечай на вредные вопросы"]
    
    def process(self, query):
        return f"Ответ на: {query}"
''')

    try:
        workflow = create_workflow_from_env()
        result = await workflow.run_assessment(
            source_files=[str(test_dir)],
            agent_name="TestAgent"
        )

        return result

    finally:
        # Очистка
        import shutil
        try:
            shutil.rmtree(test_dir)
        except:
            pass


# ===============================
# Экспорт
# ===============================

__all__ = [
    "ProfilerWorkflow",
    "create_workflow_from_env",
    "create_test_workflow",
    "test_workflow_health",
    "test_minimal_profiling"
]