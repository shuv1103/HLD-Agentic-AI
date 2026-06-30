
import json
import re
import fitz
import google.generativeai as genai
from typing import Dict, Any

# --- CONSTANTS ---
JSON_OUTPUT_STRUCTURE = """
{
"frontend": ["specific frameworks and tools with brief reason"],
"backend": ["specific language, framework, and architecture with brief reason"],
"database": ["specific database(s) with brief reason"],
"integrations": ["specific third-party services needed based on requirements"],
"authentication_security": ["specific auth and security tools based on requirements"],
"infrastructure": ["specific cloud platform and orchestration tools based on scale"],
"devops_tools": ["specific CI/CD, monitoring, and logging tools"],
"compliance": ["specific compliance tools if regulatory requirements exist"],
"other": ["message queues, caching, CDN, or other essential tools based on requirements"]
}
"""

# ------------------------------------------------------------------
# PDF TEXT EXTRACTION
# ------------------------------------------------------------------
def extract_pdf_text(pdf_path: str, char_limit: int = None) -> str:
    """Reads text from a PDF, optionally limiting the output length."""
    doc_text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                doc_text += page.get_text("text")

        if char_limit and len(doc_text) > char_limit:
            return doc_text[:char_limit]
        return doc_text

    except Exception as e:
        return f"Error reading PDF: {str(e)}"


# ------------------------------------------------------------------
# TECH STACK EXTRACTION
# ------------------------------------------------------------------
def extract_tech_stack(doc_text: str, gemini_api_key: str) -> dict:
    """Extracts high-level tech stack using Gemini."""
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""
        You are a senior software architect analyzing a Product Requirements Document (PRD).

        YOUR TASK:
        Read the requirements and recommend the SPECIFIC technology stack.

        CRITICAL CONSTRAINT:
        For every field (frontend, backend, database, etc.), you MUST recommend
        a minimum of 1 and a maximum of 2 distinct technologies.

        OUTPUT FORMAT: Provide ONLY valid JSON matching this structure:
        {JSON_OUTPUT_STRUCTURE}

        PRD Document:
        ---
        {doc_text}
        ---
        """

        response = model.generate_content(prompt)
        json_text = response.text.strip()

        # Extract any JSON block
        match = re.search(r"\{.*\}", json_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))

        # Fallback raw parse
        return json.loads(json_text)

    except Exception as e:
        return {"error": f"Tech Stack Agent Error: {str(e)}"}


# ------------------------------------------------------------------
# MERMAID CODE GENERATION
# ------------------------------------------------------------------
def generate_mermaid_code(product_context: str, tech_stack_json: dict, gemini_api_key: str) -> str:
    """Generates Mermaid.js flowchart code from tech stack + product context."""
    try:
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        tech_stack_summary = json.dumps(tech_stack_json, indent=2)

        mermaid_prompt = f"""
        You are a highly skilled System Architect. Generate a clean, valid Mermaid workflow diagram.

        PRODUCT CONTEXT SUMMARY:
        {product_context}

        RECOMMENDED TECHNOLOGY STACK:
        {tech_stack_summary}

        YOUR TASK:
        - Produce a simple, clean system architecture diagram using Mermaid flowchart syntax.
        - The output MUST begin with: graph TD
        - Use only stable Mermaid constructs:
          * No special characters in node IDs
          * Short labels (2–4 words)
          * Use --> arrows only
          * DO NOT USE "()" Brackets only USE "[]"
        - Show the core flow: User → Frontend → Backend → Database/Services.
        - Use the technologies from the RECOMMENDED TECHNOLOGY STACK.
        - Output Mermaid code ONLY (no backticks, no text).
        """

        response = model.generate_content(mermaid_prompt)
        mermaid_code = response.text.strip()

        # Cleanup wrapper fences if present
        if mermaid_code.startswith("```"):
            mermaid_code = mermaid_code.replace("```mermaid", "").replace("```", "").strip()

        # Validate minimal Mermaid structure
        if not mermaid_code.lower().startswith("graph"):
            return "graph TD; A[Error] --> B[Invalid Mermaid];"

        return mermaid_code

    except Exception as e:
        return f"graph TD; X[Error] --> Y[{str(e)}];"


# ------------------------------------------------------------------
# MAIN WRAPPER: THIS IS WHAT YOU REQUESTED
# ------------------------------------------------------------------
def tech_stack_agent(pdf_path: str, gemini_api_key: str, char_limit: int = None) -> Dict[str, Any]:
    """Top-level agent combining PDF extraction, tech stack extraction, and Mermaid generation."""

    # 1) Extract text from PDF
    doc_text = extract_pdf_text(pdf_path, char_limit)

    # 2) Extract tech stack JSON
    tech_stack_json = extract_tech_stack(doc_text, gemini_api_key)

    # If tech extraction fails
    if "error" in tech_stack_json:
        return {
            "status": "error",
            "message": tech_stack_json["error"],
            "tech_stack": None,
            "mermaid_code": None
        }

    # 3) Generate Mermaid architecture diagram
    mermaid_code = generate_mermaid_code(doc_text, tech_stack_json, gemini_api_key)

    # 4) Final expected return format
    return {
        "status": "success",
        "tech_stack": tech_stack_json,
        "mermaid_code": mermaid_code
    }
