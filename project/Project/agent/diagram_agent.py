"""
Diagram Generation Agent
"""

from typing import Dict, Any, List
from pathlib import Path

from .base_agent import BaseAgent
from state.models import HLDState, DiagramData
from utils.diagram_converter import diagram_plan_to_text
from utils.diagram_renderer import render_diagrams

class DiagramAgent(BaseAgent):
    """Agent responsible for generating diagrams from design plans"""
    
    def __init__(self):
        # Diagram agent doesn't need LLM, but inherits base functionality
        super().__init__("GEMINI_API_KEY", "gemini-2.0-flash")
    
    def get_system_prompt(self) -> str:
        """Not used for diagram generation"""
        return ""
    
    def process(self, state: HLDState) -> Dict[str, Any]:
        """Generate diagrams from domain and behavior plans"""
        stage = "diagram_generation"
        self.update_state_status(state, stage, "processing", "Generating diagrams...")
        
        try:
            # Prepare diagram plans
            diagram_plan = self._prepare_diagram_plan(state)
            
            # Convert plans to Mermaid text
            conversion_result = diagram_plan_to_text(diagram_plan)
            
            if "error" in conversion_result:
                warning_msg = f"Diagram conversion warning: {conversion_result['error']}"
                state.add_warning(warning_msg)
                class_text = "classDiagram\n"
                sequence_texts = []
            else:
                class_text = conversion_result.get("class_text", "classDiagram\n")
                sequence_texts = conversion_result.get("sequence_texts", [])
            
            # Build Mermaid map
            mermaid_map = {"diagram_class": class_text}
            for i, seq_text in enumerate(sequence_texts):
                mermaid_map[f"diagram_seq_{i+1}"] = seq_text
            
            # Set up output directory
            output_dir = self._get_output_dir(state)
            diagrams_dir = output_dir / "diagrams"
            
            # Get configuration
            config = state.config or {}
            render_images = config.get("render_images", True)
            image_format = config.get("image_format", "png")
            renderer = config.get("renderer", "kroki")
            
            # Render diagrams
            render_result = render_diagrams(
                mermaid_map=mermaid_map,
                out_dir=str(diagrams_dir),
                want_images=render_images,
                image_fmt=image_format,
                renderer=renderer,
                save_sources=True
            )
            
            # Prepare image paths for HLD embedding
            class_img_path = None
            seq_img_paths = []
            
            if render_images and "images" in render_result:
                class_img_path = render_result["images"].get("diagram_class")
                for i in range(len(sequence_texts)):
                    seq_img_paths.append(render_result["images"].get(f"diagram_seq_{i+1}"))
            
            # Create diagram data
            diagram_data = DiagramData(
                class_text=class_text,
                sequence_texts=sequence_texts,
                class_img_path=class_img_path,
                seq_img_paths=seq_img_paths,
                mermaid_map=mermaid_map
            )
            
            # Update state
            state.diagrams = diagram_data
            
            self.update_state_status(state, stage, "completed", "Diagrams generated successfully")
            
            return {
                "success": True,
                "data": diagram_data.dict(),
                "message": f"Generated {len(mermaid_map)} diagrams successfully",
                "render_result": render_result
            }
            
        except Exception as e:
            error_msg = f"Diagram generation failed: {str(e)}"
            self.update_state_status(state, stage, "failed", error=error_msg)
            state.add_error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _prepare_diagram_plan(self, state: HLDState) -> Dict[str, Any]:
        """Prepare diagram plan from state data"""
        plan = {}
        
        # Add class diagram plan from domain data
        if state.domain and state.domain.diagram_plan:
            plan.update(state.domain.diagram_plan)
        
        # Add sequence diagram plan from behavior data
        if state.behavior and state.behavior.diagram_plan:
            sequences = state.behavior.diagram_plan.get("sequences", [])
            plan["sequences"] = sequences
        
        # Add entities for attribute enrichment
        if state.domain and state.domain.entities:
            plan["entities"] = [entity.dict() for entity in state.domain.entities]
        
        return plan
    
    def _get_output_dir(self, state: HLDState) -> Path:
        """Get output directory for the current requirement"""
        base_output = Path("output")
        requirement_name = state.requirement_name or "unknown"
        return base_output / requirement_name




