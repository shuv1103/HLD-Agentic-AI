"""
PDF Extraction Node - Orchestrates PDF extraction stage
"""



from pathlib import Path
from typing import Any, Dict
from datetime import datetime

from .base_node import BaseNode
from agent.pdf_agent import PDFExtractionAgent
from state.models import HLDState, ExtractedContent

class PDFExtractionNode(BaseNode):
    def __init__(self):
        super().__init__(node_name="pdf_extraction")
        self.agent = PDFExtractionAgent()

    def execute(self,state:HLDState) -> HLDState:
        stage = self.node_name
        self.update_state_status(state,stage,"processing","Starting PDF extraction...")

        try:
            result = self.agent.process(state)
            if not result.get("success"):
                raise ValueError(result.get("error", "Unknown PDF extraction failure"))
            data = result.get("data",{})
            extracted = ExtractedContent(**data)

            self._validate_extracted_content(extracted,state)
            extracted.markdown = self._clean_markdown(extracted.markdown)
            state.extracted = extracted
            self.update_state_status(state,stage,"completed","PDF extraction successful")
            self._log_metrics(state,extracted)

            return state
        except FileNotFoundError as e:
            msg = f"PDF file not found: {str(e)}"
            self.update_state_status(state,stage,"failed",error=msg)
            state.add_error(msg)

        except ValueError as e:
            msg = f"Validation on parsing error : {str(e)}"
            self.update_state_status(state,stage,"failed",error=msg)
            state.add_error(msg)

        return state
    
    def _validate_extracted_content(self,content:ExtractedContent,state:HLDState) -> None:
        errors = []
        if not content.markdown.strip():
            errors.append("Extracted markdown is empty")
        
        meta = content.meta or {}

        for field in ("title","version","date"):
            if not meta.get(field):
                errors.append("missing metadata field: {field}")
        
        if not content.schema_version:
            content.schema_version="1.1"

        if errors:
            raise ValueError("; ".join(errors))
        
    def _clean_markdown(self,text: str) -> str:
        text = text.replace("\x00", "")
        text = text.strip()
        lines = [ln.rstrip() for ln in text.splitlines()]
        return "\n".join(lines)

    def _log_metrics(self,state:HLDState, content:ExtractedContent)->None:
        pdf_path = Path(state.pdf_path)
        pdf_size = pdf_path.stat().st_size if pdf_path.exists() else 0
        md_len = len(content.markdown)
        meta = content.meta or {}

        metrics = {
            "pdf_size_bytes": pdf_size,
            "markdown_length": md_len,
            "title":meta.get("version"),
            "date": meta.get("date"),
            "timestamp": datetime.utcnow().isoformat()
        }
        state.add_metric("pdf_extraction",metrics)