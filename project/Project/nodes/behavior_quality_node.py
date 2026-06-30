"""
Behavior and Quality Node - Analyzes use cases, NFRs, and risks
"""


from typing import Dict, Any, List
from .base_node import BaseNode
from agent.behavior_agent import BehaviorQualityAgent
from state.models import HLDState, BehaviorData

class BehaviorQualityNode(BaseNode):
    def __init__(self):
        super().__init__("behavior_quality")
        super().__init__(node_name="behavior_node")
        self.agent = BehaviorQualityAgent()

    def execute(self,state:HLDState) -> HLDState:
        self.update_state_status(state,self.node_name,"processing","Analyzing behavioral design and quality factors...")
        try:
            result = self.agent.process(state)
            if not result.get("success"):
                raise ValueError(result.get("error", "Behavior analysis failed"))
            data = BehaviorData(**result["data"])

            self._validate_use_cases(data)
            self._validate_nfrs(data)
            self._validate_risks(data)

            self._generate_diagram_plan(data)
            state.behavior = data
            self.update_state_status(state,"completed","Behavior and quality analysis completed")

            self._log_metrics(state,data)

            return state

        except Exception as e:
            self.update_state_status(state,"failed",str(e))
            state.add_error(f"BehaviorQualityNode failed {e}")
            return state
    
    def _validate_use_cases(self,data:BehaviorData):
        seen = set()
        filtered = []
        for uc in data.use_cases or []:
            uc_clean = uc.strip().capitalize()
            if uc_clean and uc_clean not in seen:
                seen.add(uc_clean)
                filtered.append(uc_clean)
        data.use_cases = filtered
    
    def _validate_nfrs(self,data:BehaviorData):
        standard_keys = ["security","reliability","performance","operability"]
        nfrs = data.nfrs or {}
        for key in standard_keys:
            nfrs.setdefault(key,[])
        for k, items in nfrs.items():
            nfrs[k] = [i.strip().capitalize() for i in items if i.strip()]
        data.nfrs = nfrs

    def _validate_risks(self,data:BehaviorData):
        seen_ids = set()
        for risk in data.risks or []:
            if not risk.id or risk.id in seen_ids:
                risk.id = f"R{len(seen_ids)+1:02d}"
            seen_ids.add(risk.id)
            risk.impact = min(max(risk.impact or 3, 1),5)
            risk.likelihood = min(max(risk.likelihood or 3,1),5)
            if not risk.mitigation:
                risk.mitigation = "Mitigation to be defined"
        data.risks = data.risks or []

    def _generate_diagram_plan(self,data:BehaviorData):
        if not data.use_cases:
            return 
        
        actors = self._extract_actors(data.use_cases)
        sequences = []
        for uc in data.use_cases:
            if len(actors) <2:
                continue
            steps = [{"from":actors[0],"to":actors[1],"message":uc}]
            sequences.append({"title":uc,"actors":actors,"steps":steps})
        data.diagram_plan = {"sequences":sequences}

    def _extract_actors(self, use_cases:List[str])->List[str]:
        common_actors = ["User","System","Admin","API"]
        for uc in use_cases:
            for actor in common_actors:
                if actor.lower() in uc.lower():
                    return [actor,"System"]
        return ["Actor","System"]

    def _log_metrics(self,state:HLDState, data:BehaviorData):
        metrics = {
            "use_case_count": len(data.use_cases or []),
            "risk_count": len(data.risks or []),
            "nfr_categories": len(data.nfrs.keys()),
            "avg_risk_severity":round(
                sum(r.impact+ r.likelihood for r in data.risks)/(2*len(data.risks))
                if data.risks else 0,2
            ),
        }
        state.add_metric("behavior_quality_metrics", metrics)
