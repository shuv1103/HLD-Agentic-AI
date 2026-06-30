"""
Authentication and Integrations Node - Analyzes security and external systems
"""
from typing import Dict,Any
from .base_node import BaseNode
from agent.auth_agent import AuthIntegrationsAgent
from state.models import HLDState,AuthenticationData,IntegrationData

class AuthIntegrationsNode(BaseNode):
    def __init__(self):
        super().__init__(node_name="auth_integrations")
        self.agent=AuthIntegrationsAgent()
    
    def execute(self,state:HLDState)->HLDState:
        stage="auth_integrations"
        self.update_state_status(state,stage,"processing","Analyzing authenticationa and integration...")

        try:
            result=self.agent.process(state)
            if not result.get("success"):
                raise ValueError(result.get("error","Agent procesing failed"))

            auth_data=result["data"].get("authentication")
            integration_data=result["data"].get("integrations",[])

            self._validate_authentication(auth_data)
            self._validate_integrations(integration_data)

            self.authentication=AuthenticationData(**auth_data)
            self.integrations=[IntegrationData(**i) for i in integration_data]

            self.update_state_status(state,stage,"completed","Authentication and Integration Analysis Completed")

            self._log_security_insights(state)
            return state
        
        except Exception as e:
            err=f"Authentication node failed:{e}"
            self.update_state_status(state,stage,"failed",error=err)
            state.add_error(err)
            return state
    
    def _validate_authentication(self,data:Dict[str,Any]):
        if not data or not data.get("actors"):
            raise ValueError("Authentication data missing or has no actors.")
        for key in ("flows","threats"):
            if not any(data.get(key,[])):
                self.logger.warning(f"Auth field '{key}' seems incomplete")
    
    def _validate_integrations(self,integrations:list):
        seen=set()
        for integ in integrations:
            sys=integ.get("system","")
            if sys in seen:
                self.logger.warning(f"Duplicate integration system: {sys}")
            seen.add(sys)
            dc=integ.get("data_contract",{})

            if not dc.get("inputs") or not dc.get("outputs"):
                self.logger.warning(f"Incomplete data contract for sysytem '{sys}")

    def _log_security_insights(self,state:HLDState):
        auth=state.authentication
        integ=state.integrations or []
        threat_count=len(auth.threats) if auth and auth.threats else 0
        self.logger.info(
            f"Security Insights:{len(auth.actors)} actors, {len(integ)} integrations,"
            f"{threat_count} threats detected."
        )
