"""
LangGraph Workflow Nodes for HLD Generation
"""

from typing import Dict, Any
from langchain_core.runnables import RunnableLambda

from state.models import HLDState
from agent import (
    PDFExtractionAgent,
    AuthIntegrationsAgent,
    DomainAPIAgent,
    BehaviorQualityAgent,
    DiagramAgent,
    OutputAgent
)

class WorkflowNodes:
    """Container for all workflow nodes"""
    
    def __init__(self):
        """Initialize all agents"""
        self.pdf_agent = PDFExtractionAgent()
        self.auth_agent = AuthIntegrationsAgent()
        self.domain_agent = DomainAPIAgent()
        self.behavior_agent = BehaviorQualityAgent()
        self.diagram_agent = DiagramAgent()
        self.output_agent = OutputAgent()
    
    def pdf_extraction_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """PDF extraction node"""
        hld_state = HLDState(**state)
        result = self.pdf_agent.process(hld_state)
        
        # Update state dict with processed state
        updated_state = hld_state.dict()
        updated_state["_node_result"] = result
        
        return updated_state
    
    def auth_integrations_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Authentication and integrations analysis node"""
        hld_state = HLDState(**state)
        result = self.auth_agent.process(hld_state)
        
        updated_state = hld_state.dict()
        updated_state["_node_result"] = result
        
        return updated_state
    
    def domain_api_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Domain and API design node"""
        hld_state = HLDState(**state)
        result = self.domain_agent.process(hld_state)
        
        updated_state = hld_state.dict()
        updated_state["_node_result"] = result
        
        return updated_state
    
    def behavior_quality_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Behavior and quality analysis node"""
        hld_state = HLDState(**state)
        result = self.behavior_agent.process(hld_state)
        
        updated_state = hld_state.dict()
        updated_state["_node_result"] = result
        
        return updated_state
    
    def diagram_generation_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Diagram generation node"""
        hld_state = HLDState(**state)
        result = self.diagram_agent.process(hld_state)
        
        updated_state = hld_state.dict()
        updated_state["_node_result"] = result
        
        return updated_state
    
    def output_composition_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Output composition node"""
        hld_state = HLDState(**state)
        result = self.output_agent.process(hld_state)
        
        updated_state = hld_state.dict()
        updated_state["_node_result"] = result
        
        return updated_state
    
    def should_continue(self, state: Dict[str, Any]) -> str:
        """Conditional routing based on state"""
        hld_state = HLDState(**state)
        
        # Check for critical errors that should stop the workflow
        if hld_state.has_errors():
            critical_stages = ["pdf_extraction"]
            for stage in critical_stages:
                if (stage in hld_state.status and 
                    hld_state.status[stage].status == "failed"):
                    return "END"
        
        # Continue to next stage based on completion
        if not hld_state.is_stage_completed("pdf_extraction"):
            return "pdf_extraction"
        elif not hld_state.is_stage_completed("auth_integrations"):
            return "auth_integrations"
        elif not hld_state.is_stage_completed("domain_api_design"):
            return "domain_api_design"
        elif not hld_state.is_stage_completed("behavior_quality"):
            return "behavior_quality"
        elif not hld_state.is_stage_completed("diagram_generation"):
            return "diagram_generation"
        elif not hld_state.is_stage_completed("output_composition"):
            return "output_composition"
        else:
            return "END"
    
    def get_node_runnables(self) -> Dict[str, RunnableLambda]:
        """Get all nodes as RunnableLambda objects for LangGraph"""
        return {
            "pdf_extraction": RunnableLambda(self.pdf_extraction_node),
            "auth_integrations": RunnableLambda(self.auth_integrations_node),
            "domain_api_design": RunnableLambda(self.domain_api_node),
            "behavior_quality": RunnableLambda(self.behavior_quality_node),
            "diagram_generation": RunnableLambda(self.diagram_generation_node),
            "output_composition": RunnableLambda(self.output_composition_node)
        }
