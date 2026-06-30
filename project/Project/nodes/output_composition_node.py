"""
Output Composition Node - Generates final HLD documentation
"""


import os 
import json 
import markdown
import shutil
import logging
from datetime import date, datetime
from typing import Any,Dict,List

from .base_node import BaseNode
from agent.output_agent import OutputAgent
from state.schema import HLDState

try:
    import matplotlib.pyplot as plt 
except ImportError:
    plt = None

class OutputCompositionNode(BaseNode):
    def __init__(self,node_name: str="output_composition", agent:Any = None):
        super().__init__(node_name=node_name)
        self.agent = OutputAgent()

    
    def execute(self,state:HLDState)->HLDState:
        output_dir = self._get_output_dir(state)
        self.logger.info(f"Composing final HLD output in: {output_dir}")

        state = self._run_with_monitoring(self.agent.process,state)

        try:
            markdown_content = self._compose_markdown(state)
            html_content = self._convert_to_html(markdown_content)
            diagrams_html = self._generate_interactive_viewer(state,output_dir)
            self._generate_visualizations(state, output_dir)

            self._save_artifacts(output_dir, markdown_content, html_content, diagrams_html, state)

            self._validate_outputs(output_dir)

            state.output = {
                "markdown":os.path.join(output_dir,"HLD.md"),
                "html": os.path.join(output_dir, "HLD.html"),
                "diagrams_html": os.path.join(output_dir, "Diagrams.html"),
            }
            state.stage_status[self.node_name] = "completed"

            self._log_output_metrics(output_dir)

        except Exception as e:
            self.logger.error(f"Output composition failed: {e}")
            state.errors.append({"node": self.node_name, "error": str(e)})
            state.stage_status[self.node_name] = "failed"
        return state

    def _compose_markdown(self,state:HLDState) -> str:
        sections = []
        def section(title: str, content: str) -> str:
            return f"## {title}\n\n{content.strip()}\n\n"

        sections.append(f"#{state.project_name} - High-Level Design\n")
        sections.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        toc = "- [Architecture Overview](#architecture-overview)\n"
        toc += "- [Domain Model](#domain-node)\n"
        toc += "- [APIs](#apis)\n"
        toc += "- [Risks](#risks)\n"
        toc += "- [Diagrams](#diagrams)\n"
        sections.append("## Table of Contents\n\n"+ toc+ "\n")

        if hasattr(state,"architecture"):
            sections.append(section("Architecture Overview", json.dumps(state.architecture,indent=2)))

        domain = getattr(state,"domain", None)
