from langgraph.graph import END, START, StateGraph

from agents.diagnosis_agent import diagnosis_agent
from agents.rag_agent import rag_agent
from agents.report_agent import report_agent
from agents.video_agent import video_agent
from core.state import ARIAState


def _or_error(next_node: str):
    def _route(state: ARIAState) -> str:
        return END if state.get("statut") == "erreur" else next_node

    return _route


# --- Phase 1 : vidéo → diagnostic → RAG ---
_analysis_builder = StateGraph(ARIAState)
_analysis_builder.add_node("video_agent", video_agent)
_analysis_builder.add_node("diagnosis_agent", diagnosis_agent)
_analysis_builder.add_node("rag_agent", rag_agent)
_analysis_builder.add_edge(START, "video_agent")
_analysis_builder.add_conditional_edges("video_agent", _or_error("diagnosis_agent"))
_analysis_builder.add_conditional_edges("diagnosis_agent", _or_error("rag_agent"))
_analysis_builder.add_conditional_edges("rag_agent", _or_error(END))
_analysis_graph = _analysis_builder.compile()

# --- Phase 2 : rapport LLM ---
_report_builder = StateGraph(ARIAState)
_report_builder.add_node("report_agent", report_agent)
_report_builder.add_edge(START, "report_agent")
_report_builder.add_conditional_edges("report_agent", _or_error(END))
_report_graph = _report_builder.compile()


async def _stream(graph, initial_state: ARIAState) -> ARIAState:
    last_state: dict = dict(initial_state)
    async for chunk in graph.astream(initial_state):
        _, node_state = next(iter(chunk.items()))
        last_state.update(node_state)
    return ARIAState(**last_state)


async def run_analysis(initial_state: ARIAState) -> ARIAState:
    return await _stream(_analysis_graph, initial_state)


async def run_report(state: ARIAState) -> ARIAState:
    return await _stream(_report_graph, state)
