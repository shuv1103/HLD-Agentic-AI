# Agentic HLD Generator

Turns a Product Requirements Document (a PRD PDF) into a complete **High-Level Design** — structured markdown/HTML, Mermaid class & sequence diagrams rendered to images, a risk heatmap, and a quality score for the generated design.

Instead of one giant prompt, HLD authoring is treated as a **six-stage agent pipeline** built on **LangGraph**. The PRD is parsed, then passed through specialised agents that each own one design concern (security, domain modelling, behaviour, diagrams), accumulating results on a shared, typed state object. The final stage composes the document and renders the diagrams. **Google Gemini** does the reasoning; an **ML (scikit-learn/XGBoost)** layer scores HLD quality.

## Tech Stack

| Layer | Primary | Notes |
| --- | --- | --- |
| **Orchestration** | LangGraph | LangChain (`core`, `community`) components for workflow wiring. |
| **LLM** | Google Gemini 2.0 Flash | Via `google-generativeai` and `langchain-google-genai`. |
| **UI** | Streamlit | Four-tab web app (`main.py`). |
| **Chat API** | FastAPI + Uvicorn | REST chat server (`server/chat_server.py`). |
| **PDF Processing** | PyMuPDF / PyPDF2 | Extracts text from PRD PDFs. |
| **Diagrams** | Mermaid via Kroki / mermaid-cli | Writes `.mmd` sources + rendered images. |
| **Machine Learning** | scikit-learn, XGBoost | With pandas, NumPy, matplotlib for training/eval. |
| **Validation** | Pydantic v2 | Typed workflow state and I/O schema validation. |

## Features

- Converts a PRD PDF into a structured, reviewable HLD (markdown + HTML) with no manual authoring.
- Derives authentication actors, flows, threats, and external integrations.
- Models domain entities, API contracts, and a class-diagram plan.
- Writes use cases, NFRs, and a risk register with a sequence-diagram plan.
- Renders class/sequence diagrams to images (Kroki or mermaid-cli) plus a risk heatmap.
- Scores the generated HLD with a rule-based scorer and trainable regression models.
- Recommends a concrete tech stack for the same PRD (separate pipeline).
- Ships a Gemini-backed chat assistant (FastAPI) embedded in the UI.

## Quick Start

**Prerequisites:** Python 3.11+, a shell, and Gemini API key(s).

```bash
# from project/Project
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` with your Gemini credentials:

```env
OPENAI/GEMINI_MODEL=your-model-choice (I used gemini)
KROKI_URL=https://kroki.io
```

Run the app (and, optionally, the chat service):

```bash
streamlit run main.py
python -m uvicorn server.chat_server:app --reload --port 8000   # optional, powers in-app chat
```

Sample PRDs live in `data/` (`Requirement-1.pdf` … `Requirement-10.pdf`). Pick one in the UI and generate; outputs are written to `output/<requirement_name>/`.

## Architecture

Five cooperating layers: an interface layer (Streamlit + FastAPI), an orchestration layer that compiles and runs a LangGraph state machine, the six-stage agent pipeline, external model/renderer dependencies, and the input PRDs plus generated artifacts.

<img width="738" height="522" alt="architecture" src="https://github.com/user-attachments/assets/a5993683-c533-40c3-996e-9d36dc8ffa9c" />

## Pipeline Lifecycle

1. User selects a PRD PDF and chooses a workflow type + diagram options.
2. `HLDWorkflow` builds an initial `HLDState` and invokes the compiled LangGraph graph.
3. **PDF Extraction** → structured markdown + metadata.
4. **Auth & Integrations** → authentication actors, flows, threats, external systems.
5. **Domain & API** → entities, request/response contracts, class-diagram plan.
6. **Behaviour & Quality** → use cases, NFRs, risk register, sequence-diagram plan.
7. **Diagram Generation** → Mermaid text + rendered images (Kroki or mermaid-cli).
8. **Output Composition** → assembles `HLD.md`, `HLD.html`, `diagrams.html`, and the risk heatmap.
9. Results/downloads render in the UI; errors and warnings collected on state are surfaced.

## The Agents

Each stage is an agent under `agent/`, all inheriting `BaseAgent`, which centralises Gemini configuration, prompt assembly, and resilient response parsing.

> **`BaseAgent` — resilient LLM calls.** `call_llm` parses output in three passes (direct `json.loads`, fenced-block extraction, first balanced brace span). If all fail and retries are enabled, it re-prompts at temperature 0 for JSON only. Every agent fixes its role and exact output schema in a system prompt and returns empty fields rather than inventing content when the PRD is silent.

Paste each agent's workflow diagram in the right-hand cell.

---

### 1. PDF Extraction Agent · `agent/pdf_agent.py`

<table>
<tr>
<td width="55%" valign="top">

- Validates the PDF path and file accessibility; stops with a clear error on a missing or corrupted file.
- Loads the full PDF content and prepares it for structured processing.
- Sends the content to the LLM with a system prompt defining its role and a JSON output format.
- Normalizes the response — parses JSON, falls back to Markdown parsing on failure, extracts the title, and fills missing metadata (version, date).
- Wraps the clean result in an `ExtractedContent` model and saves it to `state.extracted`.

</td>
<td width="45%" valign="top" align="center">

<!-- Paste PDF Extraction Agent diagram here -->
<img width="490" height="668" alt="image" src="https://github.com/user-attachments/assets/ac3831c5-51dd-46bd-97aa-68cdd79559fc" />

</td>
</tr>
</table>

---

### 2. Auth & Integrations Agent · `agent/auth_agent.py`

<table>
<tr>
<td width="55%" valign="top">

- Extracts auth and integration sections from the PRD using structured LLM prompts.
- Identifies and categorizes 4–6 key auth elements: IdPs, flows, tokens, and security threats.
- Maps 3–10 external integrations with purpose, protocols, endpoints, and data contracts.
- Normalizes output using 15+ heuristic mappings to fix missing fields, alternate keys, and formatting.
- Performs schema validation across all extracted objects, then saves results to workflow state.

</td>
<td width="45%" valign="top" align="center">

<!-- Paste Auth & Integrations Agent diagram here -->
<img width="462" height="691" alt="image" src="https://github.com/user-attachments/assets/50a7169a-1a13-4285-b846-02a298573393" />

</td>
</tr>
</table>

---

### 3. Domain & API Agent · `agent/domain_agent.py`

<table>
<tr>
<td width="55%" valign="top">

- Extracts 3–6 core domain entities from the PRD, each with 3+ validated attributes.
- Designs 3–8 API specifications with request/response schemas aligned to the domain models.
- Generates a Mermaid-ready class-diagram plan with classes and 5–15 relationships.
- Normalizes raw LLM output via multi-layer validation, enforcing `name`, `description`, `request{}`, `response{}` on every API.
- Saves the structured `DomainData` to workflow state, enabling class and sequence diagrams.

</td>
<td width="45%" valign="top" align="center">

<!-- Paste Domain & API Agent diagram here -->
<img width="512" height="682" alt="image" src="https://github.com/user-attachments/assets/79b12dad-7b8f-4953-a68b-d7131e742295" />

</td>
</tr>
</table>

---

### 4. Behaviour & Quality Agent · `agent/behavior_agent.py`

<table>
<tr>
<td width="55%" valign="top">

- Gathers PRD text and prompts for 5–8 use cases, 5–10 risks, and 4+ NFR categories (security, reliability, performance).
- Normalizes all fields into clean, ASCII-safe, workflow-ready data with a fixed NFR structure.
- Converts vague risk labels (low / medium high / critical) into a standardized 1–5 numeric scale for consistent impact × likelihood scoring.
- Auto-enriches missing fields — e.g. generating risk IDs `R01`, `R02`… when absent.
- Produces a sequence-diagram plan (>2 valid `[from] -> [to]` sequences) and saves validated `BehaviorData` to state.

</td>
<td width="45%" valign="top" align="center">

<!-- Paste Behaviour & Quality Agent diagram here -->
<img width="510" height="645" alt="image" src="https://github.com/user-attachments/assets/2367b539-5ed8-4d48-8080-c6369bb5e88f" />

</td>
</tr>
</table>

---

### 5. Diagram Agent · `agent/diagram_agent.py` · no LLM

<table>
<tr>
<td width="55%" valign="top">

- Pulls plans from the Domain and Behaviour agents, merging the class-diagram structure with 2+ sequence flows when available.
- Converts all plans into Mermaid-compatible text, with fallback logic for missing sections.
- Builds a `mermaid_map` of 1 class diagram + N sequence diagrams (2 ≤ N ≤ 5, depending on the PRD).
- Renders Mermaid to images via the configured renderer (default Kroki), producing PNG/SVG.
- Saves diagrams to disk and updates state with `DiagramData`.

</td>
<td width="45%" valign="top" align="center">

<!-- Paste Diagram Agent diagram here -->
<img width="601" height="677" alt="image" src="https://github.com/user-attachments/assets/ada1d830-8ed0-45bf-afc2-daf90a670e90" />

</td>
</tr>
</table>

---

### 6. Output Agent · `agent/output_agent.py` · no LLM

<table>
<tr>
<td width="55%" valign="top">

- Collects outputs from all upstream agents and prepares the final HLD folders (JSON, diagrams, markdown).
- Generates a risk heatmap from the Behaviour agent's impact × likelihood values and embeds it when risks exist.
- Consolidates the full HLD state — auth, integrations, domain, APIs, use cases, NFRs, risks, diagrams — into a single `HLD.md`.
- Publishes standalone `HLD.html` and `diagrams.html` by rendering the document and Mermaid diagrams together.
- Updates workflow state with output directory paths and marks completion.

</td>
<td width="45%" valign="top" align="center">

<!-- Paste Output Agent diagram here -->
<img width="386" height="637" alt="image" src="https://github.com/user-attachments/assets/151a616e-0534-492b-a3d9-db27753922ad" />

</td>
</tr>
</table>

---

### Agent Summary

| # | Agent | Produces | File |
|---|-------|----------|------|
| 1 | PDF Extraction | structured markdown + metadata | `agent/pdf_agent.py` |
| 2 | Auth & Integrations | actors, flows, IdP options, threats, systems | `agent/auth_agent.py` |
| 3 | Domain & API | entities, API contracts, class-diagram plan | `agent/domain_agent.py` |
| 4 | Behaviour & Quality | use cases, NFRs, risks, sequence-diagram plan | `agent/behavior_agent.py` |
| 5 | Diagram (no LLM) | Mermaid text + rendered images | `agent/diagram_agent.py` |
| 6 | Output (no LLM) | HLD docs, diagrams page, heatmap | `agent/output_agent.py` |

> **Node layer.** Nodes (`nodes/`) adapt agents to LangGraph: each wraps one agent, runs its stage, updates status, and returns the mutated state. `NodeManager` registers the six nodes, owns the execution order, exposes them as runnables, and provides timed sequential execution with error capture. `BaseNode` standardises logging, status updates, and the `dict ↔ HLDState` conversion at each graph boundary.

## State Model

<img width="1112" height="395" alt="image" src="https://github.com/user-attachments/assets/43c1979b-b49b-4006-80c6-ca2e4abae1e4" />

State is composed of typed Pydantic sub-models, each mirroring an agent's output contract:

| State field | Populated by | Key fields |
| --- | --- | --- |
| `extracted` (ExtractedContent) | PDF Extraction | markdown + metadata |
| `authentication` (AuthenticationData) | Auth & Integrations | `actors`, `flows`, `idp_options`, `threats` |
| `integrations` (IntegrationData) | Auth & Integrations | `system`, `purpose`, `protocol`, `auth`, `endpoints`, `data_contract` |
| `domain` (DomainData) | Domain & API | `entities[]`, `apis[]`, `diagram_plan` |
| `behavior` (BehaviorData) | Behaviour & Quality | `use_cases`, `nfrs`, `risks`, `diagram_plan` |
| `diagrams` (DiagramData) | Diagram Generation | `class_text`, `sequence_texts[]`, `image_paths`, `mermaid_map` |
| `output` (OutputData) | Output Composition | `requirement_name`, `output_dir`, `hld_path`, `diagram_path`, `heatmap_path` |
| `status` / `errors` / `warnings` | Cross-cutting | per-stage `ProcessingStatus` and run diagnostics |

## Project Structure

```
project/Project/
├── main.py                 # Streamlit app (4 tabs)
├── graph.py                # LangGraph graph builders (sequential / conditional / parallel→seq)
├── workflow/               # HLDWorkflow orchestrator + graph wiring
├── nodes/                  # one node per stage + NodeManager + BaseNode
├── agent/                  # pdf, auth, domain, behavior, diagram, output (+ BaseAgent)
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

## UI Tabs

| Tab | Function |
| --- | --- |
| **HLD Generation** | Select a PRD, configure diagrams, run the workflow, view results, download artifacts. |
| **Tech Stack Recommendation** | Run the stack recommendation pipeline for the selected PRD. |
| **ML Training** | Generate a synthetic dataset and train/evaluate HLD quality models. |
| **Quality Prediction** | Score a generated HLD with the rule-based scorer and trained ML models. |

## Auxiliary Subsystems

**Tech-Stack Predictor** — a separate LangGraph pipeline (`tech_stack_predictor/`) that reads the same PRD and recommends a concrete stack with a supporting diagram, via a base → tech → output agent chain:

1. Reads all PRD pages with PyMuPDF and extracts the content.
2. Feeds the text to the LLM with a system prompt specifying required layers (Frontend, Backend, Database, Cloud, DevOps, …), returning JSON.
3. Cleans and validates the response into a proper stack format.
4. Feeds the stack back to the LLM to generate Mermaid code for the equivalent system-architecture diagram.

**Chat Assistant** — `server/chat_server.py` is a FastAPI service exposing `POST /chat`. It keeps per-session history in memory and proxies messages to Gemini. Streamlit embeds it as an HTML component with a per-session UUID; it runs independently via uvicorn.

**Diagram & Output Utilities** — `utils/` and `diagram_publisher.py` hold the non-LLM machinery: a converter turns a diagram plan into Mermaid, the renderer writes `.mmd` sources and images via Kroki or mermaid-cli, a matplotlib routine builds the impact × likelihood heatmap, and the composer assembles the final markdown, HTML, and diagrams page.

## Quality & Evaluation

HLD quality is assessed two independent ways, because the ML layer alone isn't trustworthy enough to gate on.

**Feature extraction.** `FeatureExtractor` (`ml/models/feature_extractor.py`) derives **37** structural and semantic features from HLD markdown — counts/densities (headers, tables, code, lists), coverage flags (architecture, security, API, data-model), domain-term frequencies, and readability/consistency measures.

**ML models.** The training pipeline (`ml/training/`) generates a synthetic HLD dataset and trains regression models — `RandomForest`, `GradientBoosting`, `XGBoost` (with `SVR` and `LinearRegression` available) — reporting R², RMSE, MAE, and MAPE. Trained models are pickled and reloaded for inference.

**Rule-based scorer.** `RuleBasedQualityScorer` (`ml/models/quality_scorer.py`) runs directly on a generated HLD, scoring completeness, clarity, consistency, and security with fixed weights, returning an overall score plus missing elements and recommendations. This is the scorer the Quality Prediction tab applies to real documents.

> **Note:** the regression models are trained on a *synthetically generated* dataset. Their scores are a useful structural signal, not a validated benchmark, and not a substitute for human design review.

## Configuration

The workflow is driven by validated Pydantic schemas (`state/schema.py`):

| Schema | Fields |
| --- | --- |
| `WorkflowInput` | `pdf_path` (must end `.pdf`), `requirement_name`, `config` |
| `ConfigSchema` | `render_images`, `image_format` (`svg`\|`png`), `renderer` (`kroki`\|`mmdc`), `theme` (`default`\|`neutral`\|`dark`) |
| `WorkflowOutput` | `success`, `state`, `output_paths`, `processing_time`, `errors`, `warnings` |

| Env variable | Purpose |
| --- | --- |
| `GEMINI_MODEL` | Optional model override (default `gemini-2.0-flash`) |
| `KROKI_URL` | Optional Kroki endpoint (default `https://kroki.io`) |

## Outputs

Each run writes a self-contained package under `output/<requirement_name>/`:

- `HLD.md` and `HLD.html` — the design document.
- `diagrams.html` plus per-diagram `.mmd` sources and rendered images.
- `risk_heatmap.png` — impact × likelihood heatmap of the risk register.

Raw Mermaid sources are always written, so a renderer outage degrades to source-only output rather than failing the run.

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| LLM returns malformed / non-JSON output | Medium | Three-pass loose JSON parsing with a temperature-0 repair retry in `BaseAgent`. |
| Gemini rate limits during execution | Medium | Model configurable via env vars; agents run sequentially to spread request load. |
| Diagram renderer (Kroki) unavailable | Low | Always writes raw `.mmd` sources; degrades gracefully to source-only. |
| LLM invents details not in the PRD | Medium | Prompts prohibit fabrication and require empty fields; outputs reviewed before use. |
| Parallel execution corrupts shared state | — | Parallel topology intentionally aliased to sequential; limitation documented. |
| ML scores over-trusted | Medium | Presented as structural indicators only; paired with the rule-based scorer and human review. |

## Glossary

| Term | Definition |
| --- | --- |
| **Agentic Pipeline** | A workflow of autonomous agents, each performing one task and passing results through a shared state. |
| **HLD** | High-Level Design — an architecture doc covering components, integrations, data model, behaviour, and risks. |
| **LangGraph** | Framework for building stateful, multi-step workflows as graphs over a shared state object. |
| **HLDState** | The central typed Pydantic object shared across nodes — the single source of truth during execution. |
| **Node / Agent** | A **node** integrates an **agent** into LangGraph; the agent encapsulates the prompt, LLM call, and stage logic. |
| **Mermaid** | Text-based diagram language used for architecture diagrams, rendered via Kroki or mermaid-cli. |
| **PRD** | Product Requirements Document (PDF) — the pipeline's input. |
