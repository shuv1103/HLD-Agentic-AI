"""
Domain and API Design Node - Creates domain model and API specifications
"""

from typing import Any, Dict, List
from .base_node import BaseNode
from agent.domain_agent import DomainAPIAgent
from state.models import HLDState
from utils.diagram_renderer import render_diagrams


class DomainAPINode(BaseNode):
    """Node responsible for generating domain models and API specifications."""

    def __init__(self, node_name="domain_api_design"):
        super().__init__(node_name)
        self.agent = DomainAPIAgent()


    def _validate_entities(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Basic validation for entities."""
        errors = []
        for e in entities:
            if not e.get("name"):
                errors.append("Entity missing name.")
            attrs = e.get("attributes", [])
            if len(attrs) < 3:
                errors.append(f"Entity '{e.get('name')}' has <3 attributes.")
        return errors

    def _validate_apis(self, apis: List[Dict[str, Any]]) -> List[str]:
        """Basic validation for API specifications."""
        errors = []
        for api in apis:
            if not api.get("name"):
                errors.append("API missing name.")
            if not isinstance(api.get("request"), dict) or not api["request"]:
                errors.append(f"API '{api.get('name')}' missing valid request schema.")
            if not isinstance(api.get("response"), dict) or not api["response"]:
                errors.append(f"API '{api.get('name')}' missing valid response schema.")
        return errors

    def _infer_relationships(self, entities: List[Dict[str, Any]]) -> List[str]:
        """Infers basic entity relationships heuristically."""
        relationships = []
        for e in entities:
            for attr in e.get("attributes", []):
                if attr.endswith("_id"):
                    target = attr[:-3].capitalize()
                    relationships.append(f"{e['name']} --> {target}")
        return relationships

    def _generate_mermaid(self, entities: List[Dict[str, Any]], apis: List[Dict[str, Any]], rels: List[str]) -> str:
        """Generate a simple Mermaid class diagram."""
        lines = ["classDiagram"]
        for e in entities:
            lines.append(f"class {e['name']} {{")
            for a in e.get("attributes", []):
                lines.append(f"  +{a}")
            lines.append("}")
        lines.extend(rels)
        for api in apis:
            lines.append(f"class {api['name']}API {{\n  +request()\n  +response()\n}}")
        return "\n".join(lines)

    def execute(self, state: HLDState) -> HLDState:
        """Execute the domain & API node logic."""
        result = self._run_with_monitoring(self.agent.process, state)
        state = result if isinstance(result, HLDState) else state

        domain = getattr(state, "domain", {}) or {}
        entities, apis = domain.get("entities", []), domain.get("apis", [])

        # Validate data
        errors = self._validate_entities(entities) + self._validate_apis(apis)
        if errors:
            state.errors.extend([{"node": self.node_name, "error": e} for e in errors])

        # Generate relationships and diagram
        rels = self._infer_relationships(entities)
        mermaid_code = self._generate_mermaid(entities, apis, rels)
        rendered = render_diagrams({"domain_model": mermaid_code}, out_dir="outputs/diagrams")

        # Update state
        state.domain = {
            "entities": entities,
            "apis": apis,
            "relationships": rels,
            "diagram_mermaid": mermaid_code,
            "diagram_paths": rendered,
        }
        state.stage_status[self.node_name] = "completed" if not errors else "failed"

        # Logging insights
        self.logger.info(
            f"{self.node_name}: {len(entities)} entities, {len(apis)} APIs, "
            f"{len(rels)} relationships, {len(errors)} validation issues"
        )
        return state
