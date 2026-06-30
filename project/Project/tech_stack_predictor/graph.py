from typing import TypedDict, Any, Dict
from langgraph.graph import StateGraph,END
from tech_stack_predictor.base_agent import extract_tech_stack_from_pdf
from tech_stack_predictor.tech_stack_diagram_agent import tech_stack_agent


# 🧩 Define a State Schema for the LangGraph pipeline
class PipelineState(TypedDict, total=False):
    pdf_path: str
    base_output: Dict[str, Any]
    mermaid_output: Dict[str, Any]
    done: bool


def create_pipeline_graph(gemini_api_key: str):
    """
    Creates a LangGraph pipeline with the 3 custom agents:
    BaseAgent → ClassifierAgent → OutputAgent
    """
    # ✅ Initialize StateGraph with schema
    graph = StateGraph(state_schema=PipelineState)

    # 🧠 Base Agent Node
    def base_agent_node(state: PipelineState) -> PipelineState:
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise ValueError("PDF path not provided in state.")
        base_output = extract_tech_stack_from_pdf(pdf_path, gemini_api_key)
        state["base_output"] = base_output
        state["done"] = True 
        return state
    
    def tech_agent_node(state: PipelineState) -> PipelineState:
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise ValueError("PDF path not provided in state.")
        mermaid_output = tech_stack_agent(pdf_path, gemini_api_key)
        state["mermaid_output"] = mermaid_output
        state["done"] = True 
        return state

  

    #  Build the sequential pipeline
    graph.add_node("BaseAgent", base_agent_node)
    graph.add_node("TechAgent",tech_agent_node)


    # Connect the flow
    graph.add_edge("BaseAgent", "TechAgent")
    graph.add_edge("TechAgent",END)

    # Set start and end points
    graph.set_entry_point("BaseAgent")
    
    graph_flow = graph.compile()
    return graph_flow


