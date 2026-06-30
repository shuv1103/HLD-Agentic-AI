"""
tech_stack_app.py
Encapsulated Streamlit Tech Stack Extraction UI
"""

import streamlit as st
import streamlit_mermaid as stmd 
import os
import json
from dotenv import load_dotenv
from tech_stack_predictor.graph import create_pipeline_graph
from typing import Dict, Any

load_dotenv()


class TechStackApp:
    def __init__(self):
        """Initialize environment and directories"""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.pdf_dir = "data/"
        os.makedirs(self.pdf_dir, exist_ok=True)

        print("[DEBUG] Init success")

    # ======================================================================
    # ENTRY FUNCTION FOR EXTERNAL USE
    # ======================================================================
    def run_with_pdf(self, pdf_path: str):
        """
        Run the extraction pipeline and render results for an existing uploaded PDF.

        Args:
            pdf_path (str): Path to the uploaded PDF file.
        """
        if not pdf_path or not os.path.exists(pdf_path):
            st.error("❌ Invalid or missing PDF path.")
            return

        try:
            with st.spinner("🔄 Extracting tech stack from uploaded PDF..."):
                st.info("**Pipeline Steps:**\n1️⃣ Base Agent\n2️⃣ Tech Agent\n3️⃣ Output Agent")

                # Create compiled LangGraph pipeline
                pipeline_app = create_pipeline_graph(self.gemini_api_key)

                # Define initial state
                state = {"pdf_path": pdf_path}

                # Run the pipeline
                result_state = pipeline_app.invoke(state)

                # Display results
                if result_state.get("base_output"):
                    st.success("🎉 Tech Stack Extraction and Diagram Generation Complete!")
                    self.render_tech_stack_output(result_state)
                else:
                    st.warning("⚠️ No classified output found.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("View Error Details"):
                st.code(str(e))

    def render_tech_stack_output(self, result_state: Dict[str, Any]):
        """
        Render classified tech stack output in a beautiful UI format
        
        """
        base_output = result_state.get("base_output")
        # print("Result State: ",result_state)
        mermaid_output = result_state.get("mermaid_output", "graph TD: E[error] --> Missing mermaid code")
        # print(mermaid_output)
        st.markdown("---")
        st.markdown("## 🎯 Tech Stack Extraction Results")
        
        # Category configuration with icons and colors
        category_config = {
            "frontend": {
                "title": " Frontend Technologies",
                "icon": "",
                "color": "#667eea",
                "description": "UI frameworks, libraries, and mobile development tools"
            },
            "backend": {
                "title": "⚙️ Backend Technologies",
                "icon": "⚙️",
                "color": "#f093fb",
                "description": "Server-side frameworks, APIs, and microservices"
            },
            "database": {
                "title": "💾 Database & Storage",
                "icon": "💾",
                "color": "#4facfe",
                "description": "Databases, caching, and data storage solutions"
            },
            "integrations": {
                "title": "🔗 Integrations & APIs",
                "icon": "🔗",
                "color": "#43e97b",
                "description": "Third-party services and external integrations"
            },
            "authentication_security": {
                "title": "🔒 Authentication & Security",
                "icon": "🔒",
                "color": "#fa709a",
                "description": "Security frameworks, authentication, and encryption"
            },
            "infrastructure": {
                "title": "☁️ Infrastructure & Cloud",
                "icon": "☁️",
                "color": "#30cfd0",
                "description": "Cloud platforms, containerization, and orchestration"
            },
            "devops_tools": {
                "title": "🛠️ DevOps & CI/CD",
                "icon": "🛠️",
                "color": "#a8edea",
                "description": "CI/CD pipelines, monitoring, and deployment tools"
            },
            "compliance": {
                "title": "✅ Compliance & Governance",
                "icon": "✅",
                "color": "#fed6e3",
                "description": "KYC/AML, regulatory compliance, and audit tools"
            },
            "other": {
                "title": "📦 Other Tools & Services",
                "icon": "📦",
                "color": "#c471f5",
                "description": "Logging, monitoring, and miscellaneous tools"
            }
        }
        
        # Summary metrics
        st.markdown("### 📊 Summary Overview")
        
        total_technologies = sum(len(techs) for techs in base_output.values() if isinstance(techs, list))
        total_categories = len([cat for cat, techs in base_output.items() if isinstance(techs, list) and len(techs) > 0])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Technologies",
                value=total_technologies,
                help="Total number of technologies identified"
            )
        
        with col2:
            st.metric(
                label="Categories",
                value=total_categories,
                help="Number of technology categories"
            )
        
        with col3:
            # Find the largest category
            largest_cat = max(
                [(cat, len(techs)) for cat, techs in base_output.items() if isinstance(techs, list)],
                key=lambda x: x[1],
                default=("N/A", 0)
            )
            st.metric(
                label="Largest Category",
                value=largest_cat[0].replace("_", " ").title(),
                delta=f"{largest_cat[1]} items",
                help="Category with most technologies"
            )
        
        st.markdown("---")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Categorized View", "🔍 Detailed View","Tech Stack Diagram View"])
        
        # Tab 1: Categorized View (Cards)
        with tab1:
            st.markdown("### Technology Categories")
            
            # Iterate through categories
            for category, techs in base_output.items():
                if not isinstance(techs, list) or len(techs) == 0:
                    continue
                
                config = category_config.get(category, {
                    "title": category.replace("_", " ").title(),
                    "icon": "📌",
                    "color": "#888888",
                    "description": "Technology category"
                })
                
                # Create expandable section for each category
                with st.expander(f"{config['icon']} {config['title']} ({len(techs)} items)", expanded=False):
                    st.markdown(f"*{config['description']}*")
                    st.markdown("")
                    
                    # Display technologies in a grid-like format using columns
                    num_cols = 2
                    cols = st.columns(num_cols)
                    
                    for idx, tech in enumerate(techs):
                        col_idx = idx % num_cols
                        with cols[col_idx]:
                            # Create a card-like display for each technology
                            st.markdown(
                                f"""
                                <div style="
                                    background: linear-gradient(135deg, {config['color']}22, {config['color']}11);
                                    padding: 12px;
                                    border-radius: 8px;
                                    border-left: 4px solid {config['color']};
                                    margin-bottom: 10px;
                                ">
                                    <p style="margin: 0; font-weight: 500; color: #333;">
                                        {config['icon']} {tech}
                                    </p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
        
        # Tab 2: Detailed View (List format)
        with tab2:
            st.markdown("### Complete Technology List")
            
            for category, techs in base_output.items():
                if not isinstance(techs, list) or len(techs) == 0:
                    continue
                
                config = category_config.get(category, {
                    "title": category.replace("_", " ").title(),
                    "icon": "📌",
                    "color": "#888888",
                    "description": "Technology category"
                })
                
                st.markdown(f"#### {config['icon']} {config['title']}")
                st.markdown(f"*{config['description']}*")
                
                # Display as numbered list
                for idx, tech in enumerate(techs, 1):
                    st.markdown(f"{idx}. **{tech}**")
                
                st.markdown("")
        with tab3:
            st.subheader("Sytem Archtecture Visualization")
            stmd.st_mermaid(mermaid_output["mermaid_code"], key = "arch_diagram", height="500px")
            st.caption("Visualization created using Mermaid.js from model output")
            st.markdown("### Mermaid Code")
            st.code(mermaid_output["mermaid_code"], language="mermaid")
        


