"""
Diagram Generation Node - Converts design plans to visual diagrams
"""



import os 
import logging
from typing import Any
from .base_node import BaseNode 
from agent.diagram_agent import DiagramAgent
from state.models import HLDState, DiagramData, ProcessingStatus

logger = logging.getLogger(__name__)

class DiagramGenerationNode(BaseNode):
    def __init__(self):
        super().__init__(node_name="diagram_generation")
        self.agent = DiagramAgent()
        self.output_dir = "diagrams"
        os.makedirs(self.output_dir,exist_ok = True)

    def execute(self,state:HLDState)->HLDState:
        try:
            logger.info("Starting DiagramGenerationNode...")
            result = self.agent.process(state)

            self._validate_mermaid(result)
            self._save_and_render(result)

            state.diagrams = DiagramData(**result)
            state.status.stages["diagram_generation"] = ProcessingStatus.COMPLETED

            self._log_metrics(result)
            logger.info("Diagram generation completed successfully")
            return state
        except Exception as e:
            msg= f"Diagram generation failed: {e}"
            logger.exception(msg)
            state.status.stages["diagram_generation"] = ProcessingStatus.FAILED
            state.status.errors.append(msg)
            return state
    
    def _validate_mermaid(self,data:dict[str,Any]):
        for diag_type in ["class_diagram","sequence_diagram"]:
            text = data.get(diag_type)
            if not text or "graph" not in text and "sequenceDiagram" not in text:
                raise ValueError(f"Invalid Mermaid syntax in {diag_type}")
            if ";;" in text or text.count("{") != text.count("}"):
                raise ValueError(f"Potential Mermaid syntax issue in {diag_type}")
    
    def _save_and_render(self,data:dict[str,Any]):
        for key,content in data.items():
            if not content or not isinstance(content,str):
                continue
            mermaid_path = os.path.join(self.output_dir,f"{key}.mmd")
            with open(mermaid_path,"w",encoding="utf-8") as f:
                f.write(content.strip())
            try:
                img_path = mermaid_path.replace(".mmd",".svg")
                self._fake_render(content,img_path)
            except Exception as re:
                logger.warning(f"Rendering failed for {key}: {re}")
                continue
    
    def _fake_render(self,mermaid:str, path:str):
        with open(path,"w",encoding="utf-8") as f:
            f.write(f"<!--Rendered diagram placeholder -->\n{mermaid}")
    
    def _log_metrics(self,data: dict[str,Any]):
        for k, v in data.items():
            if isinstance(v,str):
                size = len(v)
                logger.info(f"{k}: Mermaid length = {size}")
        logger.info(f"Saved diagrams in {self.output_dir}")

