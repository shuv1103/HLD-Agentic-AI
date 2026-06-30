"""
Authentication and Integrations Agent
"""

from typing import Dict, Any, List

from .base_agent import BaseAgent
from state.models import HLDState, AuthenticationData, IntegrationData

class AuthIntegrationsAgent(BaseAgent):
    """Agent responsible for analyzing authentication and integration requirements"""
    
    def __init__(self):
        super().__init__("GEMINI_API_KEY_1", "gemini-2.0-flash")
    
    def get_system_prompt(self) -> str:
        """System prompt for authentication and integrations analysis"""
        return (
            "ROLE: Security & integrations designer.\n"
            "TASK: From the PRD, produce authentication actors/flows/idp options/threats and external integrations.\n"
            "OUTPUT: JSON ONLY (no prose, no code fences). ASCII-safe.\n"
            "SCHEMA (return exactly these keys):\n"
            "{\n"
            '  "authentication": {\n'
            '    "actors": ["string", ...],\n'
            '    "flows": ["string", ...],\n'
            '    "idp_options": ["string", ...],\n'
            '    "threats": ["string", ...]\n'
            "  },\n"
            '  "integrations": [\n'
            '    {"system":"string","purpose":"string","protocol":"string","auth":"string",\n'
            '     "endpoints":["string",...],\n'
            '     "data_contract":{"inputs":["string",...],"outputs":["string",...]}}\n'
            "  ]\n"
            "}\n"
            "NOTES:\n"
            " - Keep values concise (≤ 80 chars per item), snake_case where applicable.\n"
            " - If unknown, return empty arrays/objects (do NOT invent prose).\n"
            "CHECKLIST before you output:\n"
            " - authentication: include arrays; may be empty but present\n"
            " - integrations: include at least 3 entries if PRD allows; each has all fields (may be empty arrays)\n"
        )
    
    def process(self, state: HLDState) -> Dict[str, Any]:
        """Analyze authentication and integrations from requirements"""
        stage = "auth_integrations"
        self.update_state_status(state, stage, "processing", "Analyzing authentication and integrations...")
        
        try:
            # Prepare requirements text
            requirements_text = self.prepare_requirements_text(state)
            user_prompt = f"PRD (markdown):\n{requirements_text}\nReturn JSON only."
            
            # Call LLM
            result = self.call_llm(user_prompt, temperature=0.2, max_tokens=1800)
            
            if not result["success"]:
                error_msg = result["error"]
                self.update_state_status(state, stage, "failed", error=error_msg)
                state.add_error(error_msg)
                return result
            
            # Normalize and validate the results
            data = result["data"]
            auth_data = self._normalize_authentication(data.get("authentication", {}))
            integrations_data = self._normalize_integrations(data.get("integrations", []))
            
            # Update state
            state.authentication = auth_data
            state.integrations = integrations_data
            
            self.update_state_status(state, stage, "completed", "Authentication and integrations analysis completed")
            
            return {
                "success": True,
                "data": {
                    "authentication": auth_data.dict(),
                    "integrations": [i.dict() for i in integrations_data]
                },
                "message": "Authentication and integrations analyzed successfully"
            }
            
        except Exception as e:
            error_msg = f"Authentication/integrations analysis failed: {str(e)}"
            self.update_state_status(state, stage, "failed", error=error_msg)
            state.add_error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _normalize_authentication(self, auth_raw: Dict[str, Any]) -> AuthenticationData:
        """Normalize authentication data"""
        if not isinstance(auth_raw, dict):
            auth_raw = {}
        
        # Map common alternates
        idps = (auth_raw.get("idp_options") or 
                auth_raw.get("idps") or 
                auth_raw.get("identity_providers") or [])
        
        threats = (auth_raw.get("threats") or 
                  auth_raw.get("threat_model") or 
                  auth_raw.get("risks") or [])
        
        return AuthenticationData(
            actors=[self.normalize_string(a) for a in self.normalize_list(auth_raw.get("actors", []))],
            flows=[self.normalize_string(f) for f in self.normalize_list(auth_raw.get("flows", []))],
            idp_options=[self.normalize_string(i) for i in self.normalize_list(idps)],
            threats=[self.normalize_string(t) for t in self.normalize_list(threats)]
        )
    
    def _normalize_integrations(self, integrations_raw: List[Any]) -> List[IntegrationData]:
        """Normalize integrations data"""
        if not isinstance(integrations_raw, list):
            integrations_raw = []
        
        normalized = []
        for integration in integrations_raw:
            if not isinstance(integration, dict):
                continue
            
            # Map alternates for system name
            system = (integration.get("system") or 
                     integration.get("service") or 
                     integration.get("provider") or 
                     integration.get("name") or "")
            
            # Map alternates for purpose
            purpose = (integration.get("purpose") or 
                      integration.get("description") or 
                      integration.get("desc") or "")
            
            # Map alternates for protocol
            protocol = (integration.get("protocol") or 
                       integration.get("transport") or 
                       integration.get("pattern") or "")
            
            # Map alternates for auth
            auth = (integration.get("auth") or 
                   integration.get("auth_type") or 
                   integration.get("security") or "")
            
            # Normalize endpoints
            endpoints_raw = self.normalize_list(integration.get("endpoints", []))
            endpoints = []
            for ep in endpoints_raw:
                if isinstance(ep, dict):
                    # Extract path/name/url from endpoint dict
                    ep_str = (ep.get("path") or ep.get("name") or ep.get("url") or "")
                    if ep_str:
                        endpoints.append(self.normalize_string(ep_str))
                else:
                    ep_str = self.normalize_string(ep)
                    if ep_str:
                        endpoints.append(ep_str)
            
            # Normalize data contract
            dc_raw = integration.get("data_contract", {})
            if not isinstance(dc_raw, dict):
                dc_raw = {}
            
            inputs = dc_raw.get("inputs", [])
            outputs = dc_raw.get("outputs", [])
            
            # Handle request/response mapping to inputs/outputs
            if not inputs and "request" in dc_raw and isinstance(dc_raw["request"], dict):
                inputs = list(dc_raw["request"].keys())
            
            if not outputs and "response" in dc_raw and isinstance(dc_raw["response"], dict):
                outputs = list(dc_raw["response"].keys())
            
            data_contract = {
                "inputs": [self.normalize_string(i) for i in self.normalize_list(inputs)],
                "outputs": [self.normalize_string(o) for o in self.normalize_list(outputs)]
            }
            
            # Only add meaningful integrations
            if system:
                normalized.append(IntegrationData(
                    system=self.normalize_string(system),
                    purpose=self.normalize_string(purpose),
                    protocol=self.normalize_string(protocol),
                    auth=self.normalize_string(auth),
                    endpoints=endpoints,
                    data_contract=data_contract
                ))
        
        return normalized
