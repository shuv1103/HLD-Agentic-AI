"""
Behavior and Quality Agent
"""

import re
from typing import Dict, Any, List

from .base_agent import BaseAgent
from state.models import HLDState, BehaviorData, RiskData

class BehaviorQualityAgent(BaseAgent):
    """Agent responsible for behavior analysis and quality attributes"""
    
    def __init__(self):
        super().__init__("GEMINI_API_KEY_2", "gemini-2.0-flash")
    
    def get_system_prompt(self) -> str:
        """System prompt for behavior and quality analysis"""
        return (
            "ROLE: Senior HLD behavior & quality writer.\n"
            "TASK: From the PRD, produce behavior (use cases), NFRs, risks, and a sequence-diagram plan.\n"
            "OUTPUT: JSON ONLY (no prose, no code fences). ASCII-safe.\n"
            "SCHEMA (return exactly these keys):\n"
            "{\n"
            '  "use_cases": ["string", ...],\n'
            '  "nfrs": {\n'
            '    "security": ["string", ...],\n'
            '    "reliability": ["string", ...],\n'
            '    "performance": ["string", ...],\n'
            '    "operability": ["string", ...]\n'
            "  },\n"
            '  "risks": [\n'
            '    {\n'
            '      "id":"string",\n'
            '      "desc":"string",\n'
            '      "assumption":"string",\n'
            '      "mitigation":"string",\n'
            '      "impact": 1,\n'
            '      "likelihood": 1\n'
            '    }, ...\n'
            "  ],\n"
            '  "diagram_plan": { "sequences": [\n'
            '    {"title":"string","actors":["string",...],"steps":[{"from":"A","to":"B","message":"..."}, ...]}, ...\n'
            "  ] }\n"
            "}\n"
            "NOTES:\n"
            " - Prefer concise, actionable NFR bullets; use short imperative phrasing.\n"
            " - steps must be actor-to-actor messages suitable for Mermaid sequence diagrams.\n"
            " - impact and likelihood must be integers 1..5 (if uncertain, choose 3).\n"
            " - If unknown, return empty arrays/objects (don't invent prose).\n"
            "CHECKLIST before you output:\n"
            " - use_cases: 5–8 items if PRD allows\n"
            " - nfrs: include at least security, reliability, performance, operability arrays\n"
            " - risks: 5–10 items with id/desc/assumption/mitigation and numeric impact/likelihood (1..5)\n"
            " - diagram_plan.sequences: ≥2 sequences if PRD allows; each has steps[].\n"
        )
    
    def process(self, state: HLDState) -> Dict[str, Any]:
        """Analyze behavior and quality attributes from requirements"""
        stage = "behavior_quality"
        self.update_state_status(state, stage, "processing", "Analyzing behavior and quality attributes...")
        
        try:
            # Prepare requirements text
            requirements_text = self.prepare_requirements_text(state)
            user_prompt = f"PRD (markdown):\n{requirements_text}\nReturn JSON only."
            
            # Call LLM
            result = self.call_llm(user_prompt, temperature=0.2, max_tokens=2200)
            
            if not result["success"]:
                error_msg = result["error"]
                self.update_state_status(state, stage, "failed", error=error_msg)
                state.add_error(error_msg)
                return result
            
            # Normalize and validate the results
            data = result["data"]
            behavior_data = self._normalize_behavior_data(data)
            
            # Update state
            state.behavior = behavior_data
            
            self.update_state_status(state, stage, "completed", "Behavior and quality analysis completed")
            
            return {
                "success": True,
                "data": behavior_data.dict(),
                "message": "Behavior and quality attributes analyzed successfully"
            }
            
        except Exception as e:
            error_msg = f"Behavior/quality analysis failed: {str(e)}"
            self.update_state_status(state, stage, "failed", error=error_msg)
            state.add_error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _normalize_behavior_data(self, data: Dict[str, Any]) -> BehaviorData:
        """Normalize behavior and quality data"""
        if not isinstance(data, dict):
            data = {}
        
        # Normalize use cases
        use_cases = self._normalize_use_cases(data.get("use_cases", []))
        
        # Normalize NFRs
        nfrs = self._normalize_nfrs(data.get("nfrs", {}))
        
        # Normalize risks
        risks = self._normalize_risks(data.get("risks", []))
        
        # Normalize diagram plan
        diagram_plan = self._normalize_sequence_plan(data.get("diagram_plan", {}))
        
        return BehaviorData(
            use_cases=use_cases,
            nfrs=nfrs,
            risks=risks,
            diagram_plan=diagram_plan
        )
    
    def _normalize_use_cases(self, use_cases_raw: List[Any]) -> List[str]:
        """Normalize use cases"""
        if not isinstance(use_cases_raw, list):
            use_cases_raw = []
        
        return [self.normalize_string(uc) for uc in use_cases_raw if uc]
    
    def _normalize_nfrs(self, nfrs_raw: Dict[str, Any]) -> Dict[str, List[str]]:
        """Normalize NFRs"""
        if not isinstance(nfrs_raw, dict):
            nfrs_raw = {}
        
        # Expected categories
        target_keys = ["security", "reliability", "performance", "operability", 
                      "usability", "maintainability", "compliance"]
        
        normalized = {}
        
        # Process target categories
        for key in target_keys:
            value = nfrs_raw.get(key, [])
            normalized[key] = [self.normalize_string(item) for item in self.normalize_list(value) if item]
        
        # Add any extra categories
        for key, value in nfrs_raw.items():
            if key not in normalized:
                normalized[key] = [self.normalize_string(item) for item in self.normalize_list(value) if item]
        
        return normalized
    
    def _normalize_risks(self, risks_raw: List[Any]) -> List[RiskData]:
        """Normalize risks with impact/likelihood quantification"""
        if not isinstance(risks_raw, list):
            risks_raw = []
        
        # Level mapping for text to numeric conversion
        level_map = {
            "vlow": 1, "very low": 1, "very-low": 1, "verylow": 1, "1": 1,
            "low": 2, "2": 2,
            "medium": 3, "med": 3, "moderate": 3, "3": 3,
            "high": 4, "4": 4,
            "vhigh": 5, "very high": 5, "very-high": 5, "veryhigh": 5, "critical": 5, "5": 5
        }
        
        def to_1_5(value) -> int:
            """Convert value to 1-5 scale"""
            if isinstance(value, (int, float)):
                try:
                    return max(1, min(5, int(round(value))))
                except Exception:
                    return 3
            if isinstance(value, str):
                return level_map.get(value.strip().lower(), 3)
            return 3
        
        risks = []
        for idx, risk in enumerate(risks_raw, start=1):
            if not isinstance(risk, dict):
                continue
            
            risk_id = self.normalize_string(risk.get("id")) or f"R{idx:02d}"
            desc = self.normalize_string(risk.get("desc") or risk.get("description"))
            assumption = self.normalize_string(risk.get("assumption"))
            mitigation = self.normalize_string(risk.get("mitigation") or risk.get("mitigation_plan"))
            
            impact = to_1_5(risk.get("impact"))
            likelihood = to_1_5(risk.get("likelihood"))
            
            risks.append(RiskData(
                id=risk_id,
                desc=desc,
                assumption=assumption,
                mitigation=mitigation,
                impact=impact,
                likelihood=likelihood
            ))
        
        return risks
    
    def _normalize_sequence_plan(self, plan_raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize sequence diagram plan"""
        if not isinstance(plan_raw, dict):
            plan_raw = {}
        
        sequences_raw = plan_raw.get("sequences", [])
        if not isinstance(sequences_raw, list):
            sequences_raw = []
        
        sequences = []
        for seq in sequences_raw:
            if not isinstance(seq, dict):
                continue
            
            title = self.normalize_string(seq.get("title") or seq.get("name"))
            actors = [self.normalize_string(a) for a in self.normalize_list(seq.get("actors", [])) if a]
            
            # Normalize steps
            steps_raw = seq.get("steps", [])
            if not isinstance(steps_raw, list):
                steps_raw = seq.get("messages", []) or seq.get("flow", []) or []
            
            steps = []
            for step in self.normalize_list(steps_raw):
                normalized_step = self._normalize_step(step)
                if normalized_step:
                    steps.append(normalized_step)
            
            sequences.append({
                "title": title,
                "actors": actors,
                "steps": steps
            })
        
        return {"sequences": sequences}
    
    def _normalize_step(self, step: Any) -> Dict[str, str]:
        """Normalize a sequence step"""
        if isinstance(step, str):
            # Parse "A->B: message" format
            match = re.match(r"\s*([^\-:>]+)\s*[-]{0,1}>\s*([^:]+)\s*:\s*(.*)\s*$", step)
            if match:
                return {
                    "from": match.group(1).strip(),
                    "to": match.group(2).strip(),
                    "message": match.group(3).strip()
                }
            
            # Fallback parsing
            parts = step.split(":")
            left = parts[0] if parts else ""
            msg = parts[1] if len(parts) > 1 else ""
            
            pair = left.split("->")
            from_actor = pair[0].strip() if pair else ""
            to_actor = pair[1].strip() if len(pair) > 1 else ""
            
            return {
                "from": from_actor,
                "to": to_actor,
                "message": msg.strip()
            }
        
        elif isinstance(step, dict):
            from_actor = self.normalize_string(
                step.get("from") or step.get("source") or 
                step.get("actor") or step.get("sender")
            )
            to_actor = self.normalize_string(
                step.get("to") or step.get("target") or step.get("receiver")
            )
            message = self.normalize_string(
                step.get("message") or step.get("msg") or step.get("text")
            )
            
            return {
                "from": from_actor,
                "to": to_actor,
                "message": message
            }
        
        return {"from": "", "to": "", "message": ""}
