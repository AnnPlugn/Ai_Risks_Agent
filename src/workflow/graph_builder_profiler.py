"""
LangGraph workflow для тестирования только профилирования ИИ-агентов
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from ..utils.llm_config_manager import get_llm_config_manager
from ..models.risk_models import WorkflowState, AgentProfile
from ..agents.profiler_agent import create_profiler_from_env
from ..utils.logger import get_langgraph_logger


class RiskAssessmentWorkflow:
    """
    Workflow только для профилирования ИИ-агентов
    """

    def __init__(self):
        manager = get_llm_config_manager()
        self.llm_base_url = manager.get_base_url()
        self.llm_model = manager.get_model()
        self.profiler = create_profiler_from_env()
        self.graph_logger = get_langgraph_logger()
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        """Построение упрощённого графа только для профилирования"""
        print("🔍 ПОСТРОЕНИЕ ГРАФА ДЛЯ ПРОФИЛИРОВАНИЯ")
        workflow = StateGraph(WorkflowState)

        print("🔍 ДОБАВЛЯЕМ УЗЛЫ")
        self._add_nodes(workflow)

        print("🔍 ДОБАВЛЯЕМ РЁБРА")
        self._add_edges(workflow)

        workflow.set_entry_point("initialization")
        workflow.set_finish_point("finalization")

        print("🔍 КОМПИЛИРУЕМ ГРАФ")
        compiled_graph = workflow.compile()
        print("🔍 ГРАФ СКОМПИЛИРОВАН")
        return compiled_graph

    def _add_nodes(self, workflow: StateGraph):
        """Добавление узлов для профилирования"""
        print("🔍 ДОБАВЛЯЕМ УЗЛЫ:")

        print("  ✅ initialization")
        workflow.add_node("initialization", self._initialization_node)

        print("  ✅ profiling")
        from ..agents.profiler_agent import create_profiler_node_function
        profiler_node = create_profiler_node_function(self.profiler)
        workflow.add_node("profiling", profiler_node)

        print("  ✅ finalization")
        workflow.add_node("finalization", self._finalization_node)

        print("🔍 УЗЛЫ ДОБАВЛЕНЫ")

    def _add_edges(self, workflow: StateGraph):
        """Добавление рёбер"""
        print("🔍 ДОБАВЛЯЕМ РЁБРА:")

        print("  ✅ initialization → profiling")
        workflow.add_edge("initialization", "profiling")

        print("  ✅ profiling → finalization")
        workflow.add_edge("profiling", "finalization")

        print("  ✅ finalization → END")
        workflow.add_edge("finalization", END)

        print("🔍 РЁБРА ДОБАВЛЕНЫ")

    async def _initialization_node(self, state: WorkflowState) -> WorkflowState:
        """Инициализация workflow"""
        assessment_id = state.get("assessment_id") or f"assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.graph_logger.log_graph_start(assessment_id, "profiler_workflow")

        if not state.get("source_files"):
            state["current_step"] = "error"
            state["error_message"] = "Не предоставлены файлы для анализа"
            return state

        state.update({
            "assessment_id": assessment_id,
            "current_step": "profiling",
            "start_time": datetime.now()
        })
        return state

    async def _finalization_node(self, state: WorkflowState) -> WorkflowState:
        """Финализация профилирования"""
        assessment_id = state["assessment_id"]
        processing_time = (datetime.now() - state.get("start_time", datetime.now())).total_seconds()

        final_assessment_data = {
            "assessment_id": assessment_id,
            "agent_profile": state.get("agent_profile", {}),
            "processing_time_seconds": processing_time,
            "status": "completed" if state.get("agent_profile") else "failed"
        }

        state.update({
            "final_assessment": final_assessment_data,
            "current_step": "completed",
            "processing_time": processing_time
        })

        self.graph_logger.log_graph_completion(
            assessment_id,
            processing_time,
            1 if state.get("agent_profile") else 0
        )
        return state

    async def run_assessment(
            self,
            source_files: List[str],
            agent_name: Optional[str] = None,
            assessment_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Запуск профилирования
        """
        initial_state = WorkflowState(
            source_files=source_files,
            preliminary_agent_name=agent_name or "Unknown_Agent",
            assessment_id=assessment_id
        )

        try:
            final_state = await self.graph.ainvoke(initial_state.dict())
            return {
                "success": True,
                "assessment_id": final_state.get("assessment_id"),
                "final_assessment": final_state.get("final_assessment"),
                "processing_time": final_state.get("processing_time"),
                "current_step": final_state.get("current_step")
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "assessment_id": initial_state.assessment_id
            }


def create_workflow_from_env() -> 'RiskAssessmentWorkflow':
    """Создание workflow из переменных окружения"""
    return RiskAssessmentWorkflow()


__all__ = [
    "RiskAssessmentWorkflow",
    "create_workflow_from_env"
]