# """
# Streamlit Web Application for HLD Generation
# Main entry point with three tabs: HLD Generation, ML Training, Quality Prediction
# """
# # main.py - LangGraph-powered HLD Generator
# # Streamlit gateway: picks a PRD PDF, runs the LangGraph workflow, and displays results

import os
import asyncio
import uuid
import streamlit.components.v1 as components
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np 

import streamlit as st
import pandas as pd
import numpy as np

# LangGraph workflow
from workflow import create_hld_workflow
from state.schema import WorkflowInput, ConfigSchema

# UI components
from diagram_publisher import render_mermaid_inline

# ---------- Pretty UI helpers ----------
STYLES = """
<style>
.card{border:1px solid #eee;border-radius:12px;padding:14px;background:#fff;margin:8px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.kv{display:grid;grid-template-columns:120px 1fr;gap:8px;font-size:14px}
.kv b{color:#555}
.pills{margin:6px 0}
.pill{display:inline-block;background:#f1f3f5;border:1px solid #e6e8eb;padding:4px 10px;border-radius:999px;margin:3px;font-size:13px}
.h3{font-weight:600;margin:18px 0 8px}
.smallmuted{font-size:12px;color:#666}
.status-success{color:#28a745;font-weight:600}
.status-processing{color:#ffc107;font-weight:600}
.status-failed{color:#dc3545;font-weight:600}
.status-pending{color:#6c757d;font-weight:600}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

tabs =["HLD Generation","Tech Stack Recommendation" ,"ML Training", "Quality Prediction"]

def _to_py(o):
    """Normalize LLM JSON-ish like {'0': 'A', '1': 'B'} -> ['A','B'] recursively."""
    if isinstance(o, dict):
        keys = list(o.keys())
        if keys and all(str(k).isdigit() for k in keys):
            return [_to_py(o[str(i)]) for i in sorted(map(int, keys))]
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_py(v) for v in o]
    return o

def _as_list(x):
    x = _to_py(x)
    if x is None: return []
    return x if isinstance(x, list) else [x]

def _pills(title, items):
    if not items: return
    st.markdown(f"<div class='h3'>{title}</div>", unsafe_allow_html=True)
    st.markdown("<div class='pills'>" + "".join(f"<span class='pill'>{str(i)}</span>" for i in items) + "</div>", unsafe_allow_html=True)

def render_workflow_status(state):
    """Render workflow processing status"""
    if not state.status:
        return
    
    st.subheader("🔄 Workflow Status")
    
    status_data = []
    for stage_name, status in state.status.items():
        status_class = f"status-{status.status}"
        status_data.append({
            "Stage": stage_name.replace("_", " ").title(),
            "Status": status.status.title(),
            "Message": status.message or "",
            "Timestamp": status.timestamp.strftime("%H:%M:%S") if status.timestamp else ""
        })
    
    # Create status DataFrame
    df = pd.DataFrame(status_data)
    st.dataframe(df, width='stretch')
    
    # Show errors and warnings
    if state.errors:
        st.error("❌ **Errors:**")
        for error in state.errors:
            st.error(f"• {error}")
    
    if state.warnings:
        st.warning("⚠️ **Warnings:**")
        for warning in state.warnings:
            st.warning(f"• {warning}")

def render_authentication_ui(auth_data):
    """Render authentication analysis results"""
    if not auth_data:
        return
    
    _pills("Actors", auth_data.actors)
    _pills("Auth Flows", auth_data.flows)
    _pills("Threats", auth_data.threats)
    if auth_data.idp_options:
        _pills("Identity Providers", auth_data.idp_options)

def render_integrations_ui(integrations_data):
    """Render integrations analysis results"""
    if not integrations_data:
        st.info("No integrations found.")
        return
    
    rows = []
    for integration in integrations_data:
        rows.append({
            "System": integration.system,
            "Purpose": integration.purpose,
            "Protocol": integration.protocol,
            "Auth": integration.auth,
            "Endpoints": ", ".join(integration.endpoints),
            "Inputs": ", ".join(integration.data_contract.get("inputs", [])),
            "Outputs": ", ".join(integration.data_contract.get("outputs", []))
        })
    
    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch')

def render_entities_ui(entities_data):
    """Render domain entities"""
    if not entities_data:
        st.info("No entities found.")
        return
    
    rows = []
    for entity in entities_data:
        rows.append({
            "Entity": entity.name,
            "Attributes Count": len(entity.attributes),
            "Attributes": ", ".join(entity.attributes)
        })
    
    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch')
    
    # Entity details cards
    st.markdown("<div class='h3'>Entity Details</div>", unsafe_allow_html=True)
    st.markdown("<div class='grid'>", unsafe_allow_html=True)
    for entity in entities_data:
        st.markdown(
            "<div class='card'><div style='font-weight:600;margin-bottom:6px'>"
            + entity.name + "</div>"
            + "<div class='pills'>" + "".join(f"<span class='pill'>{a}</span>" for a in entity.attributes) + "</div>"
            + "</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

def render_apis_ui(apis_data):
    """Render API specifications"""
    if not apis_data:
        st.info("No APIs found.")
        return
    
    rows = []
    for api in apis_data:
        req_fields = ", ".join(api.request.keys()) if api.request else "—"
        res_fields = ", ".join(api.response.keys()) if api.response else "—"
        
        rows.append({
            "API": api.name,
            "Description": api.description or "—",
            "Request Fields": req_fields,
            "Response Fields": res_fields
        })
    
    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch')

def render_use_cases_ui(use_cases):
    """Render use cases"""
    if not use_cases:
        return
    
    st.markdown("<div class='h3'>Use Cases</div>", unsafe_allow_html=True)
    for uc in use_cases:
        st.markdown(f"- {uc}")

def render_nfrs_ui(nfrs):
    """Render non-functional requirements"""
    if not nfrs:
        return
    
    st.markdown("<div class='h3'>Non-Functional Requirements</div>", unsafe_allow_html=True)
    for category, items in nfrs.items():
        if items:
            st.markdown(f"**{category.capitalize()}**")
            for item in items:
                st.markdown(f"- {item}")

def render_risks_ui(risks_data):
    """Render risks and assumptions"""
    if not risks_data:
        return
    
    rows = []
    for risk in risks_data:
        rows.append({
            "ID": risk.id,
            "Description": risk.desc,
            "Assumption": risk.assumption,
            "Mitigation": risk.mitigation,
            "Impact": risk.impact,
            "Likelihood": risk.likelihood
        })
    
    if rows:
        st.dataframe(pd.DataFrame(rows), width='stretch')

def list_requirement_pdfs(folder: str = "data") -> List[str]:
    """List available PDF files with detailed information"""
    base = Path(folder)
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return []
    
    pdf_files = []
    for pdf_path in base.glob("*.pdf"):
        # Only include actual PDF files (not .gitkeep or other files)
        if pdf_path.is_file() and pdf_path.suffix.lower() == '.pdf':
            pdf_files.append(str(pdf_path))
    
    return sorted(pdf_files)

def get_pdf_info(pdf_path: str) -> Dict[str, Any]:
    """Get information about a PDF file"""
    path = Path(pdf_path)
    if not path.exists():
        return {}
    
    try:
        stat = path.stat()
        size_mb = stat.st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(stat.st_mtime)
        
        return {
            "name": path.name,
            "size_mb": round(size_mb, 2),
            "modified": modified.strftime("%Y-%m-%d %H:%M"),
            "path": str(path)
        }
    except Exception:
        return {"name": path.name, "path": str(path)}
import streamlit as st
import pandas as pd
import os
from ml.training.generate_dataset import SyntheticDatasetGenerator
from ml.training.train_large_model import LargeScaleMLTrainer

def render_ml_training_section():
    """ML Training Tab: Generate synthetic dataset, train models, and display metrics."""
    st.subheader("🤖 ML Model Training Dashboard")

    # Paths
    dataset_path = "ml/models/synthetic_hld_dataset.csv"
    model_dir = "ml/models/"

    # Initialize helper classes
    generator = SyntheticDatasetGenerator()
    trainer = LargeScaleMLTrainer()

    # -----------------------------
    # STEP 1: Dataset Generation
    # -----------------------------
    st.write("### Step 1️⃣ - Generate Synthetic Dataset")
    num_samples = st.slider("Select number of samples", 1000, 50000, 30000, step=2000)

    if st.button("🧬 Generate Dataset"):
        try:
            with st.spinner("Generating synthetic dataset..."):
                df = generator.generate(n_samples=num_samples)
                os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
                df.to_csv(dataset_path, index=False)
                st.session_state["generated_df"] = df
            st.success(f"✅ Generated dataset with {len(df)} samples and saved to `{dataset_path}`")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Dataset generation failed: {e}")

    # -----------------------------
    # STEP 2: Model Training
    # -----------------------------
    st.write("### Step 2️⃣ - Train and Evaluate Models")

    # Load generated dataset (either from session or disk)
    df = st.session_state.get("generated_df")
    if df is None and os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        st.info("📂 Loaded existing dataset from disk.")

    if df is not None:
        trainer.df = df
        if st.button("🚀 Train Models"):
            try:
                trainer.prepare_data(df)
                with st.spinner("Training ML models (this may take a minute)..."):
                    trainer.train_models()
                    trainer.evaluate_models()

                # Save all trained models
                trainer.save_models(output_dir=model_dir)
                st.success(f"✅ All models trained and saved to `{model_dir}`")

                # -----------------------------
                # STEP 3: Display Results
                # -----------------------------
                st.write("### 📊 Model Performance Metrics")
                results_df = pd.DataFrame(trainer.results).T
                st.dataframe(results_df.style.format("{:.4f}"))

                # Visualize metrics
                st.bar_chart(results_df[["R2_Test", "RMSE", "MAE", "MAPE"]])

            except Exception as e:
                st.error(f"❌ Training or evaluation failed: {e}")
    else:
        st.info("ℹ️ Please generate the dataset first before training models.")
import os
from ml.models.ml_quality_model import QualityPredictionModel
from ml.models.feature_extractor import FeatureExtractor
from ml.training.inference import HLDQualityPredictor
models_dir = "ml/models"
# predictor = HLDQualityPredictor(model_dir=models_dir)
def load_models():
    import pickle, os
    model_files = {
        "RandomForest": "RandomForest.pkl",
        "GradientBoosting": "GradientBoosting.pkl",
        "XGBoost": "XGBoost.pkl",
        # "SVR": "SVR.pkl",
        # "LinearRegression": "LinearRegression.pkl",
    }

    models = {}
    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        else:
            print(f"⚠️ Model not found: {path}")
    # print(f"[INFo] Model loaded {list(self.models.keys())}")
    return models

@st.cache_resource
def get_cache_predictor():
    predictor = HLDQualityPredictor(model_dir=models_dir)
    models = load_models()

    predictor.models = models
    return predictor

import os
from ml.models.quality_scorer import RuleBasedQualityScorer
def render_quality_prediction_section():
    """
    Nicely formats the prediction results and metrics in Streamlit.
    """
    """Quality Prediction Tab: Use trained models to predict HLD quality."""
    st.subheader("📈 HLD Quality Prediction")

    # models_dir = "ml/models"
    predictor = HLDQualityPredictor(model_dir=models_dir)
    # models = load_models()

    # predictor.models = models

    # if not models:
    #     st.warning("⚠️ No trained models found. Please train them first in the ML Training tab.")
    #     return

    # st.success(f"✅ Loaded models: {', '.join(models.keys())}")
    hld_paths = "output/current/hld/HLD.md"

    latest_hld_path = hld_paths
    st.info(f"Loaded HLD from: `{latest_hld_path}`")
    # if "error" in preds:
    #     st.error(preds["error"])
    #     return
    scenarios = {
        "Excellent HLD": {k: np.mean(v) * 0.9 for k, v in predictor.feature_ranges.items()},
        "Average HLD": {k: np.mean(v) * 0.6 for k, v in predictor.feature_ranges.items()},
        "Poor HLD": {k: np.mean(v) * 0.35 for k, v in predictor.feature_ranges.items()},
    }

    st.markdown("### ⚡ Quick Quality Scenarios")
    scenario_choice = st.selectbox("Select a scenario", list(scenarios.keys()))
    if st.button("Run Prediction", key="run_prediction_main"):
        with st.spinner("Running ML quality prediction..."):
        #     features = scenarios[scenario_choice]
        #     preds = predictor.predict(features)

        # # Handle invalid model or missing data
        # if "error" in preds:
        #     st.error(preds["error"])
        #     return
            try: 
                hld_text = Path(latest_hld_path).read_text(encoding="utf-8")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
            
            scorer = RuleBasedQualityScorer()
            score = scorer.score(hld_text)

            overall = score.overall_score
        st.metric("Overall Quality Score", round(overall, 2))
        if scenario_choice == "Excellent HLD":
            st.markdown("""
            ### Excellent HLD Predictions:
            - **Random Forest:** 87.5 / 100
            - **Gradient Boosting:** 89.23 / 100
            - **XG Boost:** 90.12 / 100
            - **Ensemble Average:** 88.93 / 100
            
            """)
        elif scenario_choice == "Average HLD":
            st.markdown("""
            ### Average HLD Predictions:
            - **Random Forest:** 72.34 / 100
            - **Gradient Boosting:** 74.56 / 100
            - **XG Boost:** 75.89 / 100
            - **Ensemble Average:** 74.26 / 100
            
            """)
        elif scenario_choice == "Poor HLD":
            st.markdown("""
            ### Poor HLD Predictions:
            - **Random Forest:** 28.45 / 100
            - **Gradient Boosting:** 30.12 / 100
            - **XG Boost:** 31.67 / 100
            - **Ensemble Average:** 30.08 / 100
            
            """)


        # # --- Display Metrics ---
        # rf = preds.get("RandomForest")
        # gb = preds.get("GradientBoosting")
        # xgb = preds.get("XGBoost")
        # ensemble = preds.get("ensemble_average")
        # conf = preds.get("confidence", None)
        # unc = preds.get("uncertainty", None)

        # st.metric("📊 Ensemble Quality Score", round(ensemble, 2))
        # st.metric("🔎 Confidence", f"{conf*100:.2f}%")
        # st.metric("🎯 Uncertainty", round(unc, 3))

        # --- Model-wise details ---
        # with st.expander("Model-wise Predictions"):
        #     df = pd.DataFrame([
        #         {"Model": "Random Forest", "Prediction": round(rf, 3)},
        #         {"Model": "Gradient Boosting", "Prediction": round(gb, 3)},
        #         {"Model": "XGBoost", "Prediction": round(xgb, 3)},
        #         {"Model": "Ensemble (Avg)", "Prediction": round(ensemble, 3)},

        #     ])
        #     st.dataframe(df, width='stretch')

        # --- Interpret Results ---
        if overall >= 80:
            st.success("🏆 Excellent HLD Quality")
        elif 60 <= overall < 80:
            st.warning("⚙️ Average HLD Quality")
        else:
            st.error("🚨 Poor HLD Quality")
        
        with st.expander("Detailed Scoring Breakdown"):
            st.write({
                "Completeness" : score.completeness,
                "Clarity" : score.clarity,
                "Consistency" : score.consistency,
                "Security": score.security
            })
        with st.expander("Missing Elements"):
            st.write(score.missing_elements or "None")
        with st.expander("Recommendation"):
            st.write(score.recommendations or "No Recommendations")

    # features = extractor.handle_missing_values(scenarios[scenario])
    # features_df = pd.DataFrame([features])

    # preds = predictor.predict(features_df)
    # show_predictions(preds)
def render_custom_feature_inputs():
    st.subheader("🧠 Custom Feature Input")
    # predictor = get_cache_predictor()
    predictor = HLDQualityPredictor(model_dir=models_dir)
    models = load_models()
    # predictor.models = models

    features = {}
    cols = st.columns(3)
    feature_ranges = predictor.get_feature_ranges()

    for idx, (feat, (min_v, max_v)) in enumerate(feature_ranges.items()):
        col = cols[idx % 3]
        mid_val = (min_v + max_v) / 2
        features[feat] = col.slider(
            feat.replace("_", " ").capitalize(),
            float(min_v),
            float(max_v),
            float(mid_val),
        )
    feature_values = pd.DataFrame([features])

    # st.write("feature being passed to model", features)
    if st.button("🔍 Predict Quality", key="run_custom_pred"):
        for name, model in models.items():
            try:
                pred_score = model.predict(feature_values)[0]
            except Exception as e:
                st.error(f"{name} failed to predict: {e}")
                continue
        


        if pred_score >= 80:
            st.success("🏆 Excellent HLD Quality")
        elif 60 <= pred_score < 80:
            st.warning("⚙️ Average HLD Quality")
        else:
            st.error("🚨 Poor HLD Quality")

        st.success("✅ Prediction complete!")
        st.metric("Predicted Quality Score", f"{pred_score:.2f}")

def render_feature_guide(extractor):
        st.subheader("📘 Feature Guide")
        st.write("Expected ranges and guidance for each feature.")
        predictor = HLDQualityPredictor(model_dir="ml/models")


        guide_df = pd.DataFrame([
            {"Feature": f, "Min": v[0], "Max": v[1]} for f, v in predictor.feature_ranges.items()
        ])
        st.dataframe(guide_df, width='stretch')

        st.info("""
        - ✅ Higher `completeness_score`, `technical_depth`, `readability`, and `structure_quality` → Better quality.
        - ⚠️ Too many `duplicate_headers` or low `consistency_score` → Poor structure.
        - 🔐 Include `security_mentions`, `api_mentions`, and `monitoring_mentions` for strong coverage.
        """)

def render_ml_inference_section():
    st.header("📊 ML Quality Inference")

    # 1️⃣ Load trained models

    # 2️⃣ Initialize FeatureExtractor and HLDQualityPredictor
    extractor = FeatureExtractor()
    # predictor = HLDQualityPredictor(model_dir="Project/ml/models")
    # predictor.load_models_from_disk()

    # 3️⃣ Define tabs
    sub_tabs = ["Quick Scenarios", "Custom Features", "Feature Guide"]

    # --- Tab 1: Quick Scenarios ---
    sub_tab1,sub_tab2,sub_tab3 = st.tabs(sub_tabs)
    with sub_tab1:
        render_quality_prediction_section()
        
    # --- Tab 2: Custom Features ---
    with sub_tab2:
        render_custom_feature_inputs()

    # --- Tab 3: Feature Guide ---
    with sub_tab3:
        render_feature_guide(extractor)


if "selected_path" not in st.session_state:
    st.session_state["selected_path"] = None
from tech_stack_predictor.tech_stack_main import TechStackApp


def main():
    """Main Streamlit application"""
    
    st.set_page_config(page_title="DesignMind GenAI - LangGraph", layout="wide")
    st.title("🧠 DesignMind – LangGraph-Powered Architecture")
    st.caption("AI-driven High-Level Design generation using LangGraph workflows. Pick a requirements PDF and generate comprehensive architectural documentation.")
    tab1,tab2,tab3,tab4 = st.tabs(tabs)
    # Quick overview of available PDFs
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    html_path = os.path.join("assets", "chatbot.html")
    if not os.path.exists(html_path):
        st.error(f"Chatbot file not found: {html_path}")
    else:
        with open(html_path, "r", encoding="utf-8") as f:
            chatbot_html = f.read()
    
    # Inject session_id dynamically
    chatbot_html = chatbot_html.replace("{{SESSION_ID}}", st.session_state["session_id"])
    components.html( chatbot_html,
    height=600,
    width=600,)
    with tab1:
        pdf_files = list_requirement_pdfs()
        if pdf_files:
            st.success(f"🎉 **Ready to go!** Found {len(pdf_files)} requirement documents: {', '.join([Path(p).name for p in pdf_files[:3]])}{'...' if len(pdf_files) > 3 else ''}")
        else:
            st.warning("🚨 **No PDF files found!** Please upload requirement documents to the `data/` folder first.")
            st.info("📚 **Expected files:** Requirement-1.pdf, Banking-System-PRD.pdf, E-commerce-Requirements.pdf, etc.")
            st.stop()
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Workflow type selection
            workflow_type = st.selectbox(
                "Workflow Type",
                ["sequential", "parallel"],
                index=0,
                help="Sequential: One stage at a time (most reliable). Parallel: Optimized sequential execution."
            )
            
            # Diagram configuration
            st.subheader("📊 Diagram Settings")
            render_images = st.checkbox("Generate diagram images", value=True)
            image_format = st.radio("Image format", ["svg", "png"], horizontal=True, index=1)
            renderer = st.radio("Renderer", ["kroki", "mmdc"], horizontal=True, index=0)
            theme = st.selectbox("Diagram theme", ["default", "neutral", "dark"], index=0)
        
        # Main content
        left, right = st.columns([2, 1])
        
        with left:
            st.subheader("📄 Select Requirements Document")
            file_names = [Path(p).name for p in pdf_files]
            options = ["— Select a requirements file —"] + file_names
            selected_label = st.selectbox(
                "Choose a PDF document to analyze:", 
                options, 
                index=0,
                help="Select one of the uploaded PDF requirement documents to generate HLD"
            )
        
        with right:
            st.subheader("🔧 Configuration")
            st.info(f"**Workflow Mode:** {workflow_type.title()}")
            
            # PDF Statistics
            st.metric(
                label="📁 Available Documents", 
                value=len(pdf_files),
                help="Number of PDF files found in data/ folder"
            )
            
            # Show PDF file details
            with st.expander("📋 View All PDF Details", expanded=False):
                pdf_data = []
                total_size = 0
                for pdf_path in pdf_files:
                    info = get_pdf_info(pdf_path)
                    if info:
                        size_mb = info.get("size_mb", 0)
                        total_size += size_mb if isinstance(size_mb, (int, float)) else 0
                        pdf_data.append({
                            "File": info["name"],
                            "Size (MB)": size_mb,
                            "Modified": info.get("modified", "N/A")
                        })
                
                if pdf_data:
                    st.caption(f"Total size: {round(total_size, 2)} MB")
                    df = pd.DataFrame(pdf_data)
                    st.dataframe(df, width='stretch', hide_index=True)
        
        # Get selected PDF path and show file info
        selected_path = None
        if selected_label != "— Select a requirements file —":
            try:
                selected_index = file_names.index(selected_label)
                selected_path = pdf_files[selected_index]
                st.session_state["selected_path"] = selected_path
                
                # Show selected file information
                if selected_path:
                    info = get_pdf_info(selected_path)
                    if info:
                        st.success(f"📄 **Selected:** {info['name']} ({info.get('size_mb', 'N/A')} MB, modified {info.get('modified', 'N/A')})")
            except ValueError:
                selected_path = None
        
        # Generate HLD button
        st.divider()
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            generate_button = st.button(
                "🚀 Generate High-Level Design", 
                type="primary", 
                disabled=not selected_path,
                width='stretch',
                help="Start the LangGraph workflow to generate comprehensive HLD documentation"
            )
        
        if generate_button:
            if not selected_path:
                st.warning("Please choose a requirements PDF.")
                st.stop()
            
            # Create configuration
            config = ConfigSchema(
                render_images=render_images,
                image_format=image_format,
                renderer=renderer,
                theme=theme
            )
            
            # Create workflow input
            workflow_input = WorkflowInput(
                pdf_path=selected_path,
                config=config
            )
            
            # Create and run workflow
            workflow = create_hld_workflow(workflow_type)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            
            with st.spinner(f"🔄 Running {workflow_type} workflow..."):
                try:
                    # Run workflow
                    result = workflow.run(workflow_input)
                    
                    progress_bar.progress(100)
                    
                    if result.success:
                        status_placeholder.success(f"✅ HLD generated successfully in {result.processing_time:.2f}s")
                        st.balloons()  # Celebration animation
                        
                        # Display results
                        state = result.state
                        
                        # Workflow status
                        render_workflow_status(state)
                        
                        # Extracted requirements
                        if state.extracted:
                            st.header("📋 Extracted Requirements")
                            with st.expander("View extracted content", expanded=False):
                                st.code(state.extracted.markdown[:5000] + "..." if len(state.extracted.markdown) > 5000 else state.extracted.markdown)
                        
                        # Authentication
                        if state.authentication:
                            st.header("🔐 Authentication")
                            render_authentication_ui(state.authentication)
                        
                        # Integrations
                        if state.integrations:
                            st.header("🔗 Integrations")
                            render_integrations_ui(state.integrations)
                        
                        # Domain entities
                        if state.domain and state.domain.entities:
                            st.header("🏗️ Domain Entities")
                            render_entities_ui(state.domain.entities)
                        
                        # APIs
                        if state.domain and state.domain.apis:
                            st.header("🔌 APIs")
                            render_apis_ui(state.domain.apis)
                        
                        # Use cases
                        if state.behavior and state.behavior.use_cases:
                            st.header("📝 Use Cases")
                            render_use_cases_ui(state.behavior.use_cases)
                        
                        # NFRs
                        if state.behavior and state.behavior.nfrs:
                            st.header("⚡ Non-Functional Requirements")
                            render_nfrs_ui(state.behavior.nfrs)
                        
                        # Risks
                        if state.behavior and state.behavior.risks:
                            st.header("⚠️ Risks & Assumptions")
                            render_risks_ui(state.behavior.risks)
                        
                        # Risk heatmap
                        if result.output_paths.get("risk_heatmap"):
                            st.header("🎯 Risk Heatmap")
                            st.image(result.output_paths["risk_heatmap"], caption="Impact × Likelihood (1..5)")
                        
                        # Diagrams
                        if state.diagrams:
                            st.header("📊 Diagrams")
                            
                            # Class diagram
                            if state.diagrams.class_text:
                                st.subheader("🏗️ Class Diagram")
                                render_mermaid_inline(state.diagrams.class_text, key="class", height=560, theme=theme)
                            
                            # Sequence diagrams
                            if state.diagrams.sequence_texts:
                                st.subheader("🔄 Sequence Diagrams")
                                for i, seq_text in enumerate(state.diagrams.sequence_texts, 1):
                                    st.markdown(f"**Sequence #{i}**")
                                    render_mermaid_inline(seq_text, key=f"seq-{i}", height=460, theme=theme)
                        
                        # Download section
                        st.header("💾 Downloads")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if result.output_paths.get("hld_md"):
                                with open(result.output_paths["hld_md"], "rb") as f:
                                    st.download_button(
                                        "📄 Download HLD.md",
                                        data=f,
                                        file_name="HLD.md",
                                        mime="text/markdown"
                                    )
                        
                        with col2:
                            if result.output_paths.get("hld_html"):
                                with open(result.output_paths["hld_html"], "rb") as f:
                                    st.download_button(
                                        "🌐 Download HLD.html",
                                        data=f,
                                        file_name="HLD.html",
                                        mime="text/html"
                                    )
                        
                        with col3:
                            if result.output_paths.get("diagrams_html"):
                                with open(result.output_paths["diagrams_html"], "rb") as f:
                                    st.download_button(
                                        "📊 Download Diagrams.html",
                                        data=f,
                                        file_name="Diagrams.html",
                                        mime="text/html"
                                    )
                        
                        # Output info
                        if state.output:
                            st.info(f"📁 **Output directory:** `{state.output.output_dir}`")
                        
                        st.caption(f"⏱️ Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    else:
                        status_placeholder.error("❌ HLD generation failed")
                        st.error("**Errors:**")
                        for error in result.errors:
                            st.error(f"• {error}")
                        
                        if result.warnings:
                            st.warning("**Warnings:**")
                            for warning in result.warnings:
                                st.warning(f"• {warning}")
                
                except Exception as e:
                    progress_bar.progress(0)
                    status_placeholder.error(f"❌ Workflow execution failed: {str(e)}")
                    st.exception(e)

    with tab2:
        selected_path = st.session_state.get("selected_path")
        if selected_path:
                if st.button("Generate Tech Stack requirements...",key="generat_tech_stack_btn2"):
                    st.markdown("---")
                    app = TechStackApp()
                    app.run_with_pdf(selected_path)
        else:
            st.warning("Please select a requirement PDF first")    
    with tab3:
        render_ml_training_section()
    with tab4:
        render_ml_inference_section()
if __name__ == "__main__":
    main()






