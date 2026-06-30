"""
Output Composition Agent
"""

from typing import Dict, Any
from pathlib import Path

from .base_agent import BaseAgent
from state.models import HLDState, OutputData
from utils.compose_output import hld_to_markdown, save_markdown
from utils.risk_heatmap import generate_risk_heatmap
from diagram_publisher import publish_diagrams

class OutputAgent(BaseAgent):
    """Agent responsible for composing final HLD outputs"""
    
    def __init__(self):
        # Output agent doesn't need LLM, but inherits base functionality
        super().__init__("GEMINI_API_KEY", "gemini-2.0-flash")
    
    def get_system_prompt(self) -> str:
        """Not used for output composition"""
        return ""
    
    def process(self, state: HLDState) -> Dict[str, Any]:
        """Compose final HLD outputs from all processed data"""
        stage = "output_composition"
        self.update_state_status(state, stage, "processing", "Composing final outputs...")
        
        try:
            # Set up output directories
            output_dir = self._get_output_dir(state)
            paths = self._ensure_output_dirs(output_dir)
            
            # Generate risk heatmap
            heatmap_path = self._generate_risk_heatmap(state, paths["hld"])
            
            # Prepare relative paths for HLD embedding
            class_img_rel = self._get_relative_path(paths["hld"], state.diagrams.class_img_path if state.diagrams else None)
            seq_imgs_rel = []
            if state.diagrams and state.diagrams.seq_img_paths:
                for img_path in state.diagrams.seq_img_paths:
                    seq_imgs_rel.append(self._get_relative_path(paths["hld"], img_path))
            
            # Compose HLD markdown
            hld_markdown = hld_to_markdown(
                requirement_name=state.requirement_name,
                prd_markdown=state.extracted.markdown if state.extracted else "",
                authentication=state.authentication.dict() if state.authentication else {},
                integrations=[i.dict() for i in state.integrations] if state.integrations else [],
                entities=[e.dict() for e in state.domain.entities] if state.domain else [],
                apis=[a.dict() for a in state.domain.apis] if state.domain else [],
                use_cases=state.behavior.use_cases if state.behavior else [],
                nfrs=state.behavior.nfrs if state.behavior else {},
                risks=[r.dict() for r in state.behavior.risks] if state.behavior else [],
                class_mermaid_text=state.diagrams.class_text if state.diagrams else "",
                sequence_mermaid_texts=state.diagrams.sequence_texts if state.diagrams else [],
                class_img=class_img_rel,
                seq_imgs=seq_imgs_rel,
                hld_base_dir=paths["hld"]
            )
            
            # Save HLD markdown
            hld_md_path = paths["hld"] / "HLD.md"
            saved_md_path = save_markdown(hld_markdown, hld_md_path)

            current_hld_dir = Path("output/current/hld")
            current_hld_dir.mkdir(parents=True, exist_ok=True)

            current_hld_path = current_hld_dir / "HLD.md"
            with current_hld_path.open("w" , encoding="utf-8") as f:
                f.write(hld_markdown)
            
            # Append risk heatmap to HLD if available
            if heatmap_path and heatmap_path.exists():
                with hld_md_path.open("a", encoding="utf-8") as f:
                    f.write("\n\n## Risk Heatmap\n")
                    f.write("Visual distribution of risks across Impact and Likelihood.\n\n")
                    f.write("![Risk Heatmap](risk_heatmap.png)\n")
            
            # Publish diagrams and create HTML outputs
            mermaid_map = state.diagrams.mermaid_map if state.diagrams else {}
            publish_result = publish_diagrams(
                mermaid_map=mermaid_map,
                out_dir=str(paths["diagrams"]),
                title=f"{state.requirement_name} – HLD Diagrams",
                theme=state.config.get("theme", "default"),
                preview=False,
                save_fullpage_html=True,
                hld_markdown=hld_markdown,
                hld_html_out_path=str(paths["hld"] / "HLD.html")
            )
            
            # Create output data
            output_data = OutputData(
                requirement_name=state.requirement_name,
                output_dir=str(output_dir),
                hld_md_path=str(saved_md_path),

                hld_md_current = "output/current/hld/HLD.md",
                
                hld_html_path=publish_result.get("hld_html"),
                diagrams_html_path=publish_result.get("full_html"),
                risk_heatmap_path=str(heatmap_path) if heatmap_path else None
            )
            
            # Update state
            state.output = output_data
            
            self.update_state_status(state, stage, "completed", "Output composition completed successfully")
            
            return {
                "success": True,
                "data": output_data.dict(),
                "message": "HLD outputs composed successfully",
                "paths": {
                    "hld_md": str(saved_md_path),
                    "hld_html": publish_result.get("hld_html"),
                    "diagrams_html": publish_result.get("full_html"),
                    "risk_heatmap": str(heatmap_path) if heatmap_path else None
                }
            }
            
        except Exception as e:
            error_msg = f"Output composition failed: {str(e)}"
            self.update_state_status(state, stage, "failed", error=error_msg)
            state.add_error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _get_output_dir(self, state: HLDState) -> Path:
        """Get output directory for the current requirement"""
        base_output = Path("output")
        requirement_name = state.requirement_name or "unknown"
        return base_output / requirement_name
    
    def _ensure_output_dirs(self, output_dir: Path) -> Dict[str, Path]:
        """Ensure output directories exist"""
        paths = {
            "base": output_dir,
            "json": output_dir / "json",
            "diagrams": output_dir / "diagrams", 
            "hld": output_dir / "hld"
        }
        
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        
        return paths
    
    def _generate_risk_heatmap(self, state: HLDState, hld_dir: Path) -> Path:
        """Generate risk heatmap visualization"""
        if not state.behavior or not state.behavior.risks:
            return None
        
        try:
            heatmap_path = hld_dir / "risk_heatmap.png"
            risks_data = [risk.dict() for risk in state.behavior.risks]
            
            generate_risk_heatmap(
                risks=risks_data,
                out_path=str(heatmap_path),
                title="Risk Heatmap (Impact × Likelihood)"
            )
            
            return heatmap_path
            
        except Exception as e:
            state.add_warning(f"Could not generate risk heatmap: {str(e)}")
            return None
    
    def _get_relative_path(self, base_dir: Path, abs_path: str) -> str:
        """Get relative path from base directory"""
        if not abs_path:
            return None
        
        try:
            import os
            return os.path.relpath(abs_path, start=str(base_dir))
        except Exception:
            return abs_path


