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


_builder = StateGraph(ARIAState)

_builder.add_node("video_agent", video_agent)
_builder.add_node("diagnosis_agent", diagnosis_agent)
_builder.add_node("rag_agent", rag_agent)
_builder.add_node("report_agent", report_agent)

_builder.add_edge(START, "video_agent")
_builder.add_conditional_edges("video_agent", _or_error("diagnosis_agent"))
_builder.add_conditional_edges("diagnosis_agent", _or_error("rag_agent"))
_builder.add_conditional_edges("rag_agent", _or_error("report_agent"))
_builder.add_conditional_edges("report_agent", _or_error(END))

aria_graph = _builder.compile()


async def run_pipeline(initial_state: ARIAState) -> ARIAState:
    result = await aria_graph.ainvoke(initial_state)
    return ARIAState(**result)
