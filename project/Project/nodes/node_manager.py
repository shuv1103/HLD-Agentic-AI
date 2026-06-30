"""
Node Manager - Central orchestration of all workflow nodes
"""
import time
from typing import Dict, Any, List, Callable

from state.models import HLDState

from .pdf_extraction_node import PDFExtractionNode
from .auth_integrations_node import AuthIntegrationsNode
from .domain_api_node import DomainAPINode
from .behavior_quality_node import BehaviorQualityNode
from .diagram_generation_node import DiagramGenerationNode 
from .output_composition_node import OutputCompositionNode


class NodeManager:
    def __init__(self):
        self.nodes = {
            "pdf_extraction" : PDFExtractionNode(),
            "auth_integrations":AuthIntegrationsNode(),
            "domain_api_design":DomainAPINode(),
            "behavior_quality": BehaviorQualityNode(),
            "diagram_generation": DiagramGenerationNode(),
            "output_composition":OutputCompositionNode(),
        }


        self.execution_order = [
            "pdf_extraction",
            "auth_integrations",
            "domain_api_design",
            "behavior_quality",
            "diagram_generation",
            "output_composition"
        ]

        if not all (n in self.nodes for n in self.execution_order):
            raise ValueError("Node initialization incomplete or invalid")



    def get_node_runnables(self) -> Dict[str , Callable[[Dict[str,Any]], Dict[str,Any]]]:
        runnables = {}

        for name,node in self.nodes.items():
            runnables[name] = lambda state_dict , n =node: n.execute(HLDState(**state_dict)).dict()
        return runnables
    
    def get_execution_order(self) -> List[str]:
        return self.execution_order


    def get_nodes_info(self) -> Dict[str,Dict[str,Any]]:
        #Tells about the node

        info = {}

        for name, node in self.nodes.items():
            info[name] = {
                "name":name,
                "class": node.__class__.__name__,
                "stage": getattr(node,"stage_name",name),
                "description": getattr(node,"__doc__","").strip(),
                "status": "initialized",
                "inputs": getattr(node,"input_schema",None),
                "outputs": getattr(node,"output_schema",None),
            }
        return info



    def get_node(self,name:str):
        if name not in self.nodes:
            raise KeyError(f"Node not found : {name}")
        return self.nodes[name]



    def execute_node(self, name: str, state: HLDState) -> HLDState:
        #Executing a node with timing and checking for error

        if name not in self.nodes:
            raise KeyError(f"Node not found : {name}")

        node = self.nodes[name]
        start = time.time()
        try:
            new_state = node.execute(state)
            elapsed = round(time.time() - start,2)
            new_state.add_metric(f"{name} completed in {elapsed}s")
            print(f"Node {name} completed in {elapsed}s")
            return new_state
        except Exception as e:
            state.add_error(f"Node {name} failed : {str(e)}")
            print(f"Node {name} failed : {str(e)}")
            return state
    
    def execute_all_sequential(self, state: HLDState) -> HLDState:
        for name in self.execution_order:
            state = self.execute_node(name, state)

            if state.status.get(name) == "failed":
                print(f"Pipeline halted due to failure in: {name}")
                break
        return state
            


    def validate_pipeline(self) -> bool:
        missing = [n for n in self.execution_order if n not in self.nodes]
        if missing:
            raise ValueError(f"Missing nodes: {' ,'.join(missing)}")
        return True



    def summarize(self) -> None:
        print("Node Manager summary")
        for i, name in enumerate(self.execution_order, 1):
            node = self.nodes[name]
            print(f" {i}. {name:25s} -> {node.__class__.__name__}")

#changes