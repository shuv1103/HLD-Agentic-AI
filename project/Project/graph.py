"""
LangGraph Workflow Graph Definition
Defines the graph structure and execution flow for HLD generation
Located at root level for easy access and graph visualization
"""



from typing import Dict, Any, List, Callable, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable
from operator import add

from nodes.node_manager import NodeManager
from state.models import HLDState

class WorkflowGraph:
    def __init__(self):
        self.manager = NodeManager()
        self.nodes = self.manager.get_node_runnables()
        self.execution_order = self.manager.get_execution_order()



    def create_sequential_workflow_graph(self) -> StateGraph:
        #Builds a sequqntial workflow graph pdf>auth>domain>behaviour>diagram>output

        graph = StateGraph(Dict[str, Any])

        for name, runnable in self.nodes.items():
            graph.add_node(name,runnable)
        
        graph.set_entry_point("pdf_extraction")

        seq = self.execution_order
        for i in range(len(seq)-1):
            graph.add_edge(seq[i], seq[i+1])
        
        graph.add_edge(seq[-1], END)
        return graph.compile()




    def create_parallel_workflow_graph(self) -> StateGraph:
        graph = StateGraph(Dict[str, Any])

        for name, runnable in self.nodes.items():
                graph.add_node(name,runnable)

        graph.set_entry_point("pdf_extraction")

        seq = self.execution_order
        for i in range(len(seq)-1):
            graph.add_edge(seq[i], seq[i+1])
        
        graph.add_edge(seq[-1], END)

        return graph.compile()
    


    def create_conditional_workflow_graph(self) -> StateGraph:
        #Building a graph with conditional routing based on node success and failure

        def route_after_pdf(state: Dict[str,Any]) -> str:
            return "auth_integrations" if state.get("success", True) else END
        
        def route_after_auth(state: Dict[str,Any]) -> str:
            return "domain_api_design" if not state.get("failed") else END
        
        def route_after_domain(state: Dict[str,Any]) -> str:
            return "behavior_quality" if not state.get("failed") else END
        
        def route_after_behavior(state: Dict[str,Any]) -> str:
            return "diagram_generation" if not state.get("failed") else END

        def route_after_diagram(state: Dict[str,Any]) -> str:
            return "output_composition" if not state.get("failed") else END

        graph = StateGraph(Dict[str, Any])
        for name, runnable in self.nodes.items():
            graph.add_node(name, runnable)
        graph.set_entry_point("pdf_extraction")

        graph.add_conditional_edges("pdf_extraction",route_after_pdf)
        graph.add_conditional_edges("auth_integrations",route_after_auth)
        graph.add_conditional_edges("domain_api_design",route_after_domain)
        graph.add_conditional_edges("behavior_quality",route_after_behavior)
        graph.add_conditional_edges("diagram_generation",route_after_diagram)
        graph.add_edge("output_composition",END)

        return graph.compile()

    def create_graph(self, graph_type: str) -> StateGraph:
        """Factory method for graph creation"""
        if graph_type == "sequential":
            return self.create_sequential_workflow_graph()
        elif graph_type == "parallel":
            return self.create_parallel_workflow_graph()
        elif graph_type == "conditional":
            return self.create_conditional_workflow_graph()
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

    def get_execution_order(self) -> List[str]:
        """Return node execution order"""
        return self.execution_order

    def get_nodes_info(self) -> Dict[str, Dict[str, Any]]:
        """Return detailed info about all nodes"""
        return self.manager.get_nodes_info()



    def visualize(self) -> str:
        """Return simple ASCII workflow representation"""
        return " -> ".join(self.execution_order + ["END"])


def create_workflow_graph() -> StateGraph:
    """Convenience: Create a sequential workflow graph"""
    return WorkflowGraph().create_sequential_workflow_graph()

def create_parallel_workflow_graph() -> StateGraph:
    """Convenience: Create a parallel workflow graph"""
    return WorkflowGraph().create_parallel_workflow_graph()

def create_conditional_workflow_graph() -> StateGraph:
    """Convenience: Create a conditional workflow graph"""
    return WorkflowGraph().create_conditional_workflow_graph()
        


