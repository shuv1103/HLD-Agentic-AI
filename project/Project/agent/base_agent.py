"""
Base Agent class for HLD generation agents
"""

import os 
import json 
import re 
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, date
from pathlib import Path

import google.generativeai as genai 
from dotenv import load_dotenv

from state.models import HLDState


class BaseAgent(ABC):
    def __init__(self,api_key_env:str, model_name:str = None):
        load_dotenv()


        self.api_key = os.getenv(api_key_env) or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        if not self.api_key:
            raise ValueError(f"Missing {api_key_env} or GEMINI_API_KEY in environment")

        genai.configure(api_key = self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        pass

    @abstractmethod
    def process(self, state:HLDState) -> Dict[str,Any]:
        """process the state and return the result"""
        pass
    


    def parse_json_loose(self,raw_text:str) -> Optional[Dict[str,Any]]:
        """robust json parsing with fallback strategies"""
        try:
            return json.loads(raw_text)
        except Exception:
            pass

        match = re.search(r"```(?:json)?(.*?)```", raw_text, re.S | re.I)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except Exception:
                pass

        match = re.search(r"\{.*\}", raw_text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return None


    def call_llm(self, user_prompt: str, temperature: float = 0.2,
        max_tokens: int = 2000, retry_on_failure: bool = True) -> Dict[str, Any]:

        system_prompt = self.get_system_prompt()

        try:
            response = self.model.generate_content(
                [{"text": system_prompt}, {"text":user_prompt}],
                generation_config = {
                    "temperature": temperature,
                    "top_p": 1.0,
                    "max_output_tokens": max_tokens
                }
            )
        
            raw_text = getattr(response,"text","") or ""
            parsed_data = self.parse_json_loose(raw_text)

            if parsed_data is not None:
                return {"success": True, "data": parsed_data, "raw": raw_text}

            if retry_on_failure:
                repair_prompt = (
                    "your previous output was not valid JSON or missing required"
                    "please re-emit JSON only with the exact schema specified. "
                    "do not include explainations or code fences."
                )

                retry_response = self.model.generate_content(
                    [{"text":system_prompt}, {"text": user_prompt}, 
                    {"text":repair_prompt}],
                    generation_config = {
                        "temperature": 0.0,
                        "top_p": 1.0,
                        "max_output_tokens": max_tokens
                    }
                )

                retry_raw = getattr(retry_response, "text", "") or ""
                retry_parsed = self.parse_json_loose(retry_raw)

                if retry_parsed is not None:
                    return { "success": True, "data": retry_parsed, "raw": retry_raw}

                return {
                    "success": False,
                    "error": "LLM returned non-JSON after retry",
                    "raw": raw_text + "\n---RETRY---\n" + retry_raw           
                }

            return {
                "success": False,
                "error": "Could not parse JSON from LLM response",
                "raw": raw_text
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM call failed: {str(e)}",
                "raw": ""
            }
    
    def prepare_requirements_text(self, state:HLDState) -> str:
        """prepare requirements text from state""" 
        if state.extracted and state.extracted.markdown:
            return state.extracted.markdown
        return "No requirements extracted yet"

    def update_state_status(self,state:HLDState, stage:str, status:str,
        message:str = None, error: str = None):
        state.update_state_status(stage, status, message, error)
    
    def normalize_string(self, value:Any) -> str:
        return str(value or "").strip()

    def normalize_list(self,value:Any) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value 
        return [value]
    
    def get_current_date(self) -> str:
        return datetime.now().date().isoformat()

    def get_current_datetime(self) -> str:
        return datetime.now().isoformat()
        
        
        
