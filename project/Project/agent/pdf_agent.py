"""
PDF Extraction Agent for converting PDFs to structured markdown
"""

import re
from pathlib import Path
from typing import Dict, Any

from .base_agent import BaseAgent
from state.models import HLDState, ExtractedContent

class PDFExtractionAgent(BaseAgent):
    """Agent responsible for extracting structured content from PDF requirements"""
    
    def __init__(self):
        super().__init__("GEMINI_API_KEY_4", "gemini-2.0-flash")
    
    def get_system_prompt(self) -> str:
        """System prompt for PDF extraction"""
        return (
            "ROLE: Extract structured content from a PDF PRD.\n"
            "OUTPUT: JSON ONLY (no prose, no code fences). ASCII-safe.\n"
            "SCHEMA (return exactly these keys):\n"
            "{\n"
            '  "markdown": "### <title>\\n... full markdown ...",\n'
            '  "meta": {"title":"string","version":"1.0","date":"YYYY-MM"}\n'
            "}\n"
            "NOTES:\n"
            " - Keep markdown clean (headings, lists, tables). No diagrams.\n"
            " - If uncertain, include content as markdown text; do not invent details.\n"
        )
    
    def process(self, state: HLDState) -> Dict[str, Any]:
        """Extract PDF content and update state"""
        stage = "pdf_extraction"
        self.update_state_status(state, stage, "processing", "Extracting PDF content...")
        
        try:
            # Validate PDF path
            pdf_path = Path(state.pdf_path)
            if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
                error_msg = f"Invalid PDF path: {state.pdf_path}"
                self.update_state_status(state, stage, "failed", error=error_msg)
                state.add_error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Read PDF bytes
            try:
                pdf_bytes = pdf_path.read_bytes()
            except Exception as e:
                error_msg = f"Failed to read PDF: {str(e)}"
                self.update_state_status(state, stage, "failed", error=error_msg)
                state.add_error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Call LLM with PDF content
            result = self.model.generate_content(
                [
                    {"text": self.get_system_prompt()},
                    {"mime_type": "application/pdf", "data": pdf_bytes}
                ],
                generation_config={
                    "temperature": 0.2,
                    "top_p": 1.0,
                    "max_output_tokens": 1500
                }
            )
            
            raw_text = getattr(result, "text", "") or ""
            parsed_data = self.parse_json_loose(raw_text)
            
            # Handle parsing results
            if parsed_data is None:
                # If model returned plain markdown, wrap it
                md_guess = raw_text.strip()
                if md_guess:
                    title = self._extract_title_from_markdown(md_guess)
                    parsed_data = {
                        "markdown": md_guess,
                        "meta": {
                            "title": title,
                            "version": "1.0", 
                            "date": self.get_current_date()[:7]  # YYYY-MM format
                        }
                    }
                else:
                    error_msg = "Could not parse expected JSON or markdown from PDF"
                    self.update_state_status(state, stage, "failed", error=error_msg)
                    state.add_error(error_msg)
                    return {"success": False, "error": error_msg, "raw": raw_text}
            
            # Validate and normalize the extracted data
            extracted_content = self._normalize_extracted_data(parsed_data, pdf_path, raw_text)
            
            # Update state
            state.extracted = extracted_content
            self.update_state_status(state, stage, "completed", "PDF extraction completed successfully")
            
            return {
                "success": True,
                "data": extracted_content.dict(),
                "message": "PDF extracted successfully"
            }
            
        except Exception as e:
            error_msg = f"PDF extraction failed: {str(e)}"
            self.update_state_status(state, stage, "failed", error=error_msg)
            state.add_error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _extract_title_from_markdown(self, markdown: str) -> str:
        """Extract title from markdown content"""
        # Look for first ATX heading
        match = re.search(r"^\s*#{1,6}\s+(.+)$", markdown, re.M)
        if match:
            return match.group(1).strip()[:120]
        
        # Fallback to first non-empty line
        for line in markdown.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:120]
        
        return "Extracted Requirements"
    
    def _normalize_extracted_data(self, data: Dict[str, Any], pdf_path: Path, raw_text: str) -> ExtractedContent:
        """Normalize and validate extracted data"""
        markdown = data.get("markdown", "")
        meta = data.get("meta", {})
        
        # Handle alternative content field
        if not markdown and "content" in data:
            markdown = data["content"]
        
        if not markdown:
            markdown = raw_text  # Use raw text as fallback
        
        # Ensure meta has required fields
        if not isinstance(meta, dict):
            meta = {}
        
        if not meta.get("title"):
            meta["title"] = self._extract_title_from_markdown(markdown)
        
        if not meta.get("version"):
            meta["version"] = "1.0"
        
        if not meta.get("date"):
            meta["date"] = self.get_current_date()[:7]  # YYYY-MM format
        
        return ExtractedContent(
            markdown=markdown,
            meta=meta,
            schema_version="1.1",
            generated_at=self.get_current_date(),
            source={"path": str(pdf_path.resolve())}
        )
