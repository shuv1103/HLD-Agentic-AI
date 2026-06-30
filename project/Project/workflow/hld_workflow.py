

"""
Main HLD Workflow Implementation
"""

import time
from typing import Dict, Any, Optional, AsyncIterator
from pathlib import Path

from langchain_core.runnables import Runnable

from state.models import HLDState
from state.schema import WorkflowInput, WorkflowOutput, ConfigSchema, create_initial_state
from .graph import create_workflow_graph, create_parallel_workflow_graph, create_conditional_workflow_graph

class HLDWorkflow:
    """Main workflow orchestrator for HLD generation"""
    
    def __init__(self, workflow_type: str = "sequential"):
        """
        Initialize HLD workflow
        
        Args:
            workflow_type: Type of workflow ("sequential", "parallel", "conditional")
        """
        self.workflow_type = workflow_type
        self.graph = self._create_graph()
    
    def _create_graph(self) -> Runnable:
        """Create the appropriate workflow graph"""
        if self.workflow_type == "parallel":
            # Use optimized sequential execution to avoid LangGraph concurrent update issues
            return create_parallel_workflow_graph()  # This is now optimized sequential
        elif self.workflow_type == "conditional":
            return create_conditional_workflow_graph()
        else:
            return create_workflow_graph()
    
    def run(self, input_data: WorkflowInput) -> WorkflowOutput:
        """
        Run the HLD generation workflow
        
        Args:
            input_data: Workflow input containing PDF path and configuration
            
        Returns:
            WorkflowOutput with results and metadata
        """
        start_time = time.time()
        
        try:
            # Create initial state
            initial_state = create_initial_state(input_data.pdf_path, input_data.config)
            
            # Run the workflow
            final_state_dict = self.graph.invoke(initial_state.dict())
            
            # Convert back to HLDState
            final_state = HLDState(**final_state_dict)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare output paths
            output_paths = {}
            if final_state.output:
                output_paths = {
                    "hld_md": final_state.output.hld_md_path,
                    "hld_html": final_state.output.hld_html_path,
                    "diagrams_html": final_state.output.diagrams_html_path,
                    "risk_heatmap": final_state.output.risk_heatmap_path
                }
            
            return WorkflowOutput(
                success=not final_state.has_errors(),
                state=final_state,
                output_paths=output_paths,
                processing_time=processing_time,
                errors=final_state.errors,
                warnings=final_state.warnings
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Workflow execution failed: {str(e)}"
            
            return WorkflowOutput(
                success=False,
                state=HLDState(pdf_path=input_data.pdf_path),
                processing_time=processing_time,
                errors=[error_msg]
            )
    
    async def arun(self, input_data: WorkflowInput) -> WorkflowOutput:
        """
        Run the workflow asynchronously
        
        Args:
            input_data: Workflow input
            
        Returns:
            WorkflowOutput with results
        """
        start_time = time.time()
        
        try:
            # Create initial state
            initial_state = create_initial_state(input_data.pdf_path, input_data.config)
            
            # Run the workflow asynchronously
            final_state_dict = await self.graph.ainvoke(initial_state.dict())
            
            # Convert back to HLDState
            final_state = HLDState(**final_state_dict)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare output paths
            output_paths = {}
            if final_state.output:
                output_paths = {
                    "hld_md": final_state.output.hld_md_path,
                    "hld_html": final_state.output.hld_html_path,
                    "diagrams_html": final_state.output.diagrams_html_path,
                    "risk_heatmap": final_state.output.risk_heatmap_path
                }
            
            return WorkflowOutput(
                success=not final_state.has_errors(),
                state=final_state,
                output_paths=output_paths,
                processing_time=processing_time,
                errors=final_state.errors,
                warnings=final_state.warnings
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Async workflow execution failed: {str(e)}"
            
            return WorkflowOutput(
                success=False,
                state=HLDState(pdf_path=input_data.pdf_path),
                processing_time=processing_time,
                errors=[error_msg]
            )
    
    async def stream(self, input_data: WorkflowInput) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream workflow execution with real-time updates
        
        Args:
            input_data: Workflow input
            
        Yields:
            State updates during workflow execution
        """
        try:
            # Create initial state
            initial_state = create_initial_state(input_data.pdf_path, input_data.config)
            
            # Stream the workflow execution
            async for state_update in self.graph.astream(initial_state.dict()):
                yield state_update
                
        except Exception as e:
            yield {
                "error": f"Workflow streaming failed: {str(e)}",
                "timestamp": time.time()
            }
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """Get information about the workflow configuration"""
        return {
            "workflow_type": self.workflow_type,
            "nodes": [
                "pdf_extraction",
                "auth_integrations", 
                "domain_api_design",
                "behavior_quality",
                "diagram_generation",
                "output_composition"
            ],
            "supports_parallel": self.workflow_type in ["parallel", "conditional"],
            "supports_streaming": True
        }

def create_hld_workflow(workflow_type: str = "sequential") -> HLDWorkflow:
    """
    Factory function to create HLD workflow
    
    Args:
        workflow_type: Type of workflow to create
        
    Returns:
        Configured HLD workflow instance
    """
    return HLDWorkflow(workflow_type=workflow_type)

# Convenience functions for different workflow types
def create_sequential_workflow() -> HLDWorkflow:
    """Create a sequential workflow"""
    return create_hld_workflow("sequential")

def create_parallel_workflow() -> HLDWorkflow:
    """Create a parallel workflow"""
    return create_hld_workflow("parallel")

def create_conditional_workflow() -> HLDWorkflow:
    """Create a conditional workflow"""
    return create_hld_workflow("conditional")

#Final HLD Workflow
