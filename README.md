# Agentic HLD Generator

This project turns a Product Requirements Document (a PRD PDF) into a complete High-Level Design: structured markdown and HTML, Mermaid class and sequence diagrams rendered to images, a risk heatmap, and a quality score for the generated design. It's built as a six-stage agent pipeline on **LangGraph**, with **Google Gemini** doing the reasoning and  *ML(scikit-learn)* layer for scoring HLD quality.

The idea is to treat HLD authoring as a staged pipeline rather than one big prompt. A requirements PDF is parsed, then passed through a sequence of specialised agents that each own one design concern — security, domain modelling, behaviour, diagrams — accumulating their results on a shared, typed state object. The final stage composes the design document and renders the diagrams.

## Tech Stack
| Layer                  | Primary                         | Notes                                                                                   |
| ---------------------- | ------------------------------- | --------------------------------------------------------------------------------------- |
| **Orchestration**      | LangGraph               | Uses LangChain (`core` and `community`) components for workflow orchestration.    |
| **LLM**                | Google Gemini (2.0 Flash)       | Integrated through `google-generativeai` and `langchain-google-genai`.                  |
| **UI**                 | Streamlit                       | Four-tab web application implemented in `main.py`.                                      |
| **Chat API**           | FastAPI + Uvicorn               | REST API server implemented in `server/chat_server.py`.                                 |
| **PDF Processing**     | PyMuPDF / PyPDF2                | Extracts text from PRD PDF documents.                                                   |
| **Diagram Generation** | Mermaid via Kroki / mermaid-cli | Produces Mermaid (`.mmd`) source files and rendered diagram images.                     |
| **Machine Learning**   | scikit-learn, XGBoost           | Uses `pandas`, `NumPy`, and `matplotlib` for data processing, training, and evaluation. |
| **Validation**         | Pydantic v2                     | Provides typed workflow state management and input/output schema validation.            |



## Feartures

- Converts a PRD PDF into a structured, reviewable HLD (markdown + HTML) with no manual authoring.
- Derives authentication actors, flows, threats, and external integrations.
- Models domain entities, API contracts, and a class-diagram plan.
- Writes use cases, non-functional requirements, and a risk register with a sequence-diagram plan.
- Renders class and sequence diagrams to images (Kroki or mermaid-cli) and a risk heatmap.
- Scores the generated HLD with both a rule-based scorer and trainable regression models.
- Recommends a concrete tech stack for the same PRD (separate pipeline).
- Ships a Gemini-backed chat assistant (FastAPI) embedded in the UI.

## Quick start

**Prerequisites:** Python 3.11+, a running shell, and Gemini API key(s).

```bash
# from project/Project
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` with your Gemini keys. The pipeline reads a separate key per agent; point them all at one key to start:

```env
GEMINI_API_KEY=your_key
GEMINI_API_KEY_1=your_key
GEMINI_API_KEY_2=your_key
GEMINI_API_KEY_3=your_key
GEMINI_API_KEY_4=your_key
# optional
GEMINI_MODEL=gemini-2.0-flash
KROKI_URL=https://kroki.io
```

Run the app:

```bash
streamlit run main.py
```

Run the chat service (optional — powers the in-app assistant):

```bash
python -m uvicorn server.chat_server:app --reload --port 8000
```

Sample PRDs live in `data/` (`Requirement-1.pdf` … `Requirement-10.pdf`). Pick one in the UI and generate; outputs are written under `output/<requirement_name>/`.

## Architecture

The system is organised into five cooperating layers: an interface layer (Streamlit + FastAPI), an orchestration layer that compiles and runs a LangGraph state machine, the six-stage agent pipeline, the external model and renderer dependencies, and the input PRDs and generated artifacts.

```
<img width="738" height="522" alt="image" src="https://github.com/user-attachments/assets/a5993683-c533-40c3-996e-9d36dc8ffa9c" />

```

## Pipeline lifecycle

A single HLD generation run proceeds as follows:

1. The user selects a requirements PDF and chooses a workflow type and diagram options.
2. `HLDWorkflow` builds an initial `HLDState` and invokes the compiled LangGraph graph.
3. **PDF Extraction** converts the PRD into structured markdown and metadata.
4. **Auth & Integrations** derives authentication actors, flows, threats, and external systems.
5. **Domain & API** extracts entities, request/response contracts, and a class-diagram plan.
6. **Behaviour & Quality** produces use cases, NFRs, a risk register, and a sequence-diagram plan.
7. **Diagram Generation** converts the plans to Mermaid and renders images (Kroki or mermaid-cli).
8. **Output Composition** assembles `HLD.md`, `HLD.html`, `diagrams.html`, and the risk heatmap, and returns output paths.
9. Results and downloads are rendered in the UI; errors and warnings collected on the state are surfaced.

## The agents

Each design stage is implemented by an agent under `agent/`. All agents inherit `BaseAgent`, which centralises Gemini configuration, prompt assembly, and resilient response parsing. Each LLM agent reads its own key so per-stage rate limits stay independent.

| # | Agent | Produces | Gemini key | File |
|---|-------|----------|-----------|------|
| 1 | PDF Extraction | structured markdown + metadata | `KEY_4` | `agent/pdf_agent.py` |
| 2 | Auth & Integrations | actors, flows, IdP options, threats, systems | `KEY_1` | `agent/auth_agent.py` |
| 3 | Domain & API | entities, API contracts, class-diagram plan | `KEY_3` | `agent/domain_agent.py` |
| 4 | Behaviour & Quality | use cases, NFRs, risks, sequence-diagram plan | `KEY_2` | `agent/behavior_agent.py` |
| 5 | Diagram | Mermaid text + rendered images | — (no LLM) | `agent/diagram_agent.py` |
| 6 | Output | HLD docs, diagrams page, heatmap | — (no LLM) | `agent/output_agent.py` |

**`BaseAgent` — resilient LLM calls.** `call_llm` wraps Gemini with defensive parsing. Output is parsed in three passes (direct `json.loads`, fenced-block extraction, first balanced brace span). If all fail and retries are enabled, the agent re-prompts at temperature 0 asking for JSON only, then parses again. Every agent also fixes its role and exact output schema in a system prompt and is instructed to return empty fields rather than invent content when the PRD is silent.

**Node layer.** Nodes (`nodes/`) adapt agents to LangGraph: each node wraps one agent, runs its stage, updates per-stage status, and returns the mutated state. `NodeManager` (`nodes/node_manager.py`) registers the six nodes, owns the canonical execution order, exposes them as LangGraph runnables, and provides timed sequential execution with error capture. `BaseNode` standardises logging, status updates, and the `dict ↔ HLDState` conversion at each graph boundary.

## State model

```
<img width="1037" height="635" alt="image" src="https://github.com/user-attachments/assets/3a32f185-ec03-4a2e-97d5-35fe1b4d0af9" />

```

State is composed of typed Pydantic sub-models, each mirroring an agent's output contract:

| State field | Populated by |
|-------------|--------------|
| `extracted` (ExtractedContent) | PDF Extraction — markdown + meta |
| `authentication` / `integrations` | Auth & Integrations |
| `domain` (DomainData) | Domain & API — `entities`, `apis`, `diagram_plan` |
| `behavior` (BehaviorData) | Behaviour & Quality — `use_cases`, `nfrs`, `risks`, `diagram_plan` |
| `diagrams` (DiagramData) | Diagram Generation — class/sequence Mermaid + image paths |
| `output` (OutputData) | Output Composition — file paths for HLD / diagrams / heatmap |
| `status` / `errors` / `warnings` | Cross-cutting — per-stage `ProcessingStatus` and run diagnostics |

## Project structure

```
project/Project/
├── main.py                 # Streamlit app (4 tabs)
├── graph.py                # LangGraph graph builders (sequential / conditional / parallel→seq)
├── workflow/               # HLDWorkflow orchestrator + graph wiring
├── nodes/                  # one node per stage + NodeManager + BaseNode
├── agent/                  # agents: pdf, auth, domain, behavior, diagram, output (+ BaseAgent)
├── state/                  # Pydantic state (models.py) + I/O schemas (schema.py)
├── ml/
│   ├── models/             # feature_extractor (37 features), quality_scorer (rule-based)
│   └── training/           # synthetic dataset generation + model training
├── tech_stack_predictor/   # separate LangGraph pipeline for stack recommendation
├── server/chat_server.py   # FastAPI chat endpoint
├── utils/                  # diagram convert/render, risk heatmap, output composition
├── diagram_publisher.py    # diagrams.html assembly
└── data/                   # sample requirement PDFs
```


## Data Architecture

**1. Data Ingestion -** Requirement PDFs live in data/ (sample set Requirement-1.pdf … Requirement-10.pdf). The PDF Extraction stage reads the document with PyMuPDF and asks Gemini to emit clean markdown plus a small metadata block (title, version, date). The agent is instructed not to invent details and to fall back to plain markdown when uncertain.
**2. State Schema Models-** The state is composed of typed Pydantic sub-models, each mirroring an agent's output contract:
| Model                       | Key fields                                                                                                               |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **AuthenticationData**      | `actors`, `flows`, `idp_options`, `threats`                                                                              |
| **IntegrationData**         | `system`, `purpose`, `protocol`, `auth`, `endpoints`, `data_contract`                                                    |
| **EntityData / APIData**    | **EntityData:** `name`, `attributes`<br>**APIData:** `name`, `description`, `request`, `response`                        |
| **DomainData**              | `entities[]`, `apis[]`, `diagram_plan`                                                                                   |
| **RiskData / BehaviorData** | **RiskData:** `id`, `desc`, `assumption`, `mitigation`<br>**BehaviorData:** `use_cases`, `nfrs`, `risks`, `diagram_plan` |
| **DiagramData**             | `class_text`, `sequence_texts[]`, `image_paths`, `mermaid_map`                                                           |
| **OutputData**              | `requirement_name`, `output_dir`, `hld_path`, `diagram_path`, `heatmap_path`                                             |


**3. Results -** Each run writes a self-contained package under output/<requirement_name>/: HLD.md and HLD.html (the design document), diagrams.html and per-diagram .mmd sources with rendered images, and risk_heatmap.png. Raw Mermaid sources are always written, so a renderer outage degrades to source-only output rather than failing the run.

## Workflow Architecture
The workflow is driven by validated Pydantic schemas (state/schema.py):
| Schema             | Fields                                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| **WorkflowInput**  | `pdf_path` *(must end with `.pdf`)*, `requirement_name`, `config`                                                        |
| **ConfigSchema**   | `render_images`, `image_format` (`svg` | `png`), `renderer` (`kroki` | `mmdc`), `theme` (`default` | `neutral` | `dark`) |
| **WorkflowOutput** | `success`, `state`, `output_paths`, `processing_time`, `errors`, `warnings`                                              |


## UI Capabilities
| Tab                           | Function                                                                                            |
| ----------------------------- | --------------------------------------------------------------------------------------------------- |
| **HLD Generation**            | Select a PRD, configure diagrams, run the workflow, view generated results, and download artifacts. |
| **Tech Stack Recommendation** | Run the technology stack recommendation pipeline for the selected PRD.                              |
| **ML Training**               | Generate a synthetic dataset and train/evaluate HLD quality prediction models.                      |
| **Quality Prediction**        | Evaluate a generated HLD using the rule-based quality scorer and trained machine learning models.   |



## Auxiliary subsystems

**Tech-stack predictor -** A separate LangGraph pipeline under `tech_stack_predictor/` reads the same PRD and recommends a concrete technology stack with a supporting diagram, via a small base → tech → output agent chain. Surfaced as its own UI tab.

**Chat assistant -** `server/chat_server.py` is a FastAPI service exposing `POST /chat`. It keeps per-session history in memory and proxies messages to Gemini. The Streamlit app embeds it as an HTML component with a per-session UUID; it runs independently via uvicorn.

**Diagram & output utilities.** `utils/` and `diagram_publisher.py` hold the non-LLM machinery: a converter turns a diagram plan into Mermaid, the renderer writes `.mmd` sources and renders images via Kroki or mermaid-cli, a matplotlib routine builds the impact × likelihood risk heatmap, and the composer assembles the final markdown, HTML, and diagrams page.

## Quality & evaluation

HLD quality is assessed two independent ways, because the ML layer alone wasn't trustworthy enough to gate on.

**Feature extraction.** `FeatureExtractor` (`ml/models/feature_extractor.py`) derives **37** structural and semantic features from HLD markdown — counts and densities (headers, tables, code, lists), coverage flags (architecture, security, API, data-model sections), domain-term frequencies, and readability/consistency measures.

**ML models.** The training pipeline (`ml/training/`) generates a synthetic HLD dataset and trains regression models — `RandomForest`, `GradientBoosting`, and `XGBoost` (with `SVR` and `LinearRegression` available) — reporting R², RMSE, MAE, and MAPE. Trained models are pickled and reloaded for inference.

**Rule-based scorer.** `RuleBasedQualityScorer` (`ml/models/quality_scorer.py`) runs directly on a generated HLD and scores completeness, clarity, consistency, and security with fixed weights, returning an overall score plus missing elements and recommendations. This is the scorer the Quality Prediction tab applies to real generated documents.

> **Note on the ML layer:** the regression models are trained on a *synthetically generated* dataset. Their scores are a useful structural signal, not a validated benchmark, and are not a substitute for human design review.


## Risks & Mitigration
| Risk                                              | Likelihood | Mitigation                                                                                                                                                                        |
| ------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLM returns malformed or non-JSON output**      | Medium     | Uses three-pass loose JSON parsing with a temperature-0 repair retry implemented in `BaseAgent`.                                                                                  |
| **Gemini rate limits during workflow execution**  | Medium     | Assigns a distinct API key per agent; model selection and API keys are configurable through environment variables.                                                                |
| **Diagram renderer (Kroki) unavailable**          | Low        | Always generates raw Mermaid (`.mmd`) source files; workflow degrades gracefully to source-only output.                                                                           |
| **LLM invents details not present in the PRD**    | Medium     | Prompts explicitly prohibit fabrication, require empty fields when information is unavailable, and outputs are expected to be reviewed before use.                                |
| **True parallel execution corrupts shared state** | —          | Parallel workflow topology is intentionally aliased to sequential execution and this limitation is documented.                                                                    |
| **Machine learning scores are over-trusted**      | Medium     | ML scores are presented as structural quality indicators only; real HLDs are additionally evaluated using the rule-based scorer, with human review remaining part of the process. |



## Configuration

The workflow is driven by validated Pydantic schemas (`state/schema.py`):

| Schema | Fields |
|--------|--------|
| `WorkflowInput` | `pdf_path` (must end `.pdf`), `requirement_name`, `config` |
| `ConfigSchema` | `render_images`, `image_format` (`svg`\|`png`), `renderer` (`kroki`\|`mmdc`), `theme` (`default`\|`neutral`\|`dark`) |
| `WorkflowOutput` | `success`, `state`, `output_paths`, `processing_time`, `errors`, `warnings` |

| Env variable | Purpose |
|--------------|---------|
| `GEMINI_API_KEY` | Base key (diagram / output / chat / tech-stack) |
| `GEMINI_API_KEY_1..4` | Per-agent keys (auth, behaviour, domain, pdf) |
| `GEMINI_MODEL` | Optional model override (default `gemini-2.0-flash`) |
| `KROKI_URL` | Optional Kroki endpoint (default `https://kroki.io`) |

## Outputs

Each run writes a self-contained package under `output/<requirement_name>/`:

- `HLD.md` and `HLD.html` — the design document.
- `diagrams.html` plus per-diagram `.mmd` sources and rendered images.
- `risk_heatmap.png` — impact × likelihood heatmap of the risk register.


## Summary

| Term                                    | Definition                                                                                                                                               |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agentic Pipeline**                    | A workflow composed of autonomous agents, where each agent performs a specific task and passes results through a shared workflow state.                  |
| **HLD (High-Level Design)**             | An architecture document describing the system's components, integrations, data model, behavior, and associated risks.                                   |
| **LangGraph**                           | A framework for building stateful, multi-step workflows as graphs that operate on a shared state object.                                                 |
| **HLDState**                            | The central typed Pydantic object shared across all workflow nodes, serving as the single source of truth throughout execution.                          |
| **Node / Agent**                        | A **node** integrates an **agent** into the LangGraph workflow. The agent encapsulates the prompt, LLM interaction, and stage-specific processing logic. |
| **Mermaid**                             | A text-based diagram definition language used to generate architecture diagrams, rendered via Kroki or `mermaid-cli`.                                    |
| **RuleBasedQualityScorer**              | A model-free evaluation component that scores an HLD based on structural quality metrics such as completeness, clarity, consistency, and security.       |
| **PRD (Product Requirements Document)** | The input requirements document (PDF) from which the pipeline generates the High-Level Design.                                                           |

