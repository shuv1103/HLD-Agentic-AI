"""
Domain and API Design Agent
"""
from typing import Dict, Any, List
from .base_agent import BaseAgent
from state.models import HLDState, DomainData, EntityData, APIData

class DomainAPIAgent(BaseAgent):
    #THIS AGENT IS RESPONSILBLE FOR DOMAIN MODELING AND API DESIGN

        def __init__(self):
            super().__init__("GEMINI_API_KEY_3","gemini-2.0-flash")
        
        def get_system_prompt(self) -> str:
            return (
            "ROLE: Senior domain & API designer.\n"
            "TASK: Extract domain entities, API specs, and a class-diagram plan from the PRD.\n"
            "OUTPUT: JSON ONLY (no prose, no code fences). ASCII-safe.\n"
            "SCHEMA (return exactly these keys):\n"
            "{\n"
            '  "entities": [ {"name": "string", "attributes": ["string", ...]}, ... ],\n'
            '  "apis": [ {"name": "string", "description": "string", "request": {"field":"type", ...}, "response": {"field":"type", ...}}, ... ],\n'
            '  "diagram_plan": { "class": { "classes": ["string", ...], "relationships": ["string", ...] } }\n'
            "}\n"
            "NOTES:\n"
            " - Keep attribute and field names short, snake_case.\n"
            " - relationships are Mermaid-friendly lines like: 'A --> B : uses'.\n"
            " - If unknown, return empty array/object (don't invent prose).\n"
            " - Use ≤ 60 chars for class names and relationship labels.\n"
            "CHECKLIST before you output:\n"
            " - entities: ≥ 3 with ≥ 3 attributes each (if PRD allows)\n"
            " - apis: ≥ 3 entries; each has name, description, request{}, response{}\n"
            " - diagram_plan.class: classes[] and relationships[] reference your entities/APIs\n"
        )

        def process(self, state: HLDState) -> Dict[str, Any]:
            #To design domain entities and APIs from requirements
            stage = "domain_api_design"
            self.update_state_status(state, stage, "processing", "Designing domain entities and APIs...")
            
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
                domain_data = self._normalize_domain_data(data)
                
                # Update state
                state.domain = domain_data
                
                self.update_state_status(state, stage, "completed", "Domain and API design completed")
                
                return {
                    "success": True,
                    "data": domain_data.dict(),
                    "message": "Domain entities and APIs designed successfully"
                }
                
            except Exception as e:
                error_msg = f"Domain/API design failed: {str(e)}"
                self.update_state_status(state, stage, "failed", error=error_msg)
                state.add_error(error_msg)
                return {"success": False, "error": error_msg}
    
            
        

        def _normalize_domain_data(self,data: Dict[str, Any]) -> DomainData:
            #Normalize domain and API Data

            if not isinstance(data,dict):
                data = {}

            entities = self._normalize_entities(data.get("entities", []))

            #Normalize APIs
            apis = self._normalize_apis(data.get("apis", []))

            #normalize diagram plan
            diagram_plan = self._normalize_diagram_plan(data.get("diagram_plan", {}))

            return DomainData(
                    entities=entities,
                    apis=apis,
                    diagram_plan = diagram_plan
                )


        def _normalize_entities(self, entities_raw: List[Any]) -> List[EntityData]:
            #Normalize enitity data
            if not isinstance(entities_raw,list):
                entities_raw =[]
            
            entities = []
            for entity in entities_raw:
                if not isinstance(entity,dict):
                    continue
                
                name = self.normalize_string(entity.get("name"))
                if not name:
                    continue

                attributes_raw = entity.get("attributes",[])
                if not isinstance(attributes_raw,list):
                    attributes_raw = []
                
                attributes = [self.normalize_string(attr) for attr in attributes_raw if attr]

                entities.append(EntityData(name=name,attributes=attributes))
            return entities
        


        def _normalize_apis(self,apis_raw: List[Any]) -> List[APIData]:
            if not isinstance(apis_raw, list):
                apis_raw = []
        
            apis = []
            for api in apis_raw:
                if not isinstance(api, dict):
                    continue
                
                name = self.normalize_string(api.get("name"))
                if not name:
                    continue
                
                description = self.normalize_string(api.get("description"))
                
                request = api.get("request", {})
                if not isinstance(request, dict):
                    request = {}
                
                response = api.get("response", {})
                if not isinstance(response, dict):
                    response = {}
                
                apis.append(APIData(
                    name=name,
                    description=description,
                    request=request,
                    response=response
                ))
            
            return apis


        
        def _normalize_diagram_plan(self, plan_raw: Dict[str, Any]) -> Dict[str, Any]:
            """Normalize class diagram plan"""
            if not isinstance(plan_raw, dict):
                plan_raw = {}
            
            class_plan = plan_raw.get("class", {})
            if not isinstance(class_plan, dict):
                class_plan = {}
            
            # Get classes (support multiple field names)
            classes = class_plan.get("classes")
            if not isinstance(classes, list):
                classes = (class_plan.get("nodes") or 
                          plan_raw.get("nodes") or 
                          plan_raw.get("classes") or [])
            
            if not isinstance(classes, list):
                classes = []
            
            # Normalize classes to strings
            normalized_classes = []
            for cls in classes:
                if isinstance(cls, dict):
                    name = cls.get("name", "")
                    if name:
                        normalized_classes.append(self.normalize_string(name))
                else:
                    cls_str = self.normalize_string(cls)
                    if cls_str:
                        normalized_classes.append(cls_str)
            
            # Get relationships
            relationships = class_plan.get("relationships")
            if not isinstance(relationships, list):
                relationships = (class_plan.get("relations") or 
                               plan_raw.get("relations") or 
                               plan_raw.get("relationships") or [])
            
            if not isinstance(relationships, list):
                relationships = []
            
            # Normalize relationships to strings
            normalized_relationships = []
            for rel in relationships:
                rel_str = self.normalize_string(rel)
                if rel_str:
                    normalized_relationships.append(rel_str)
            
            return {
                "class": {
                    "classes": normalized_classes,
                    "relationships": normalized_relationships
                }
            }



