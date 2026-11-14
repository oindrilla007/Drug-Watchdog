# Drug Interaction Watchdog

A multi-agent clinical AI system that reads a patient's medications, predicts dangerous drug-drug interactions, and explains why they matter for that patient, with a citation behind every claim.

Two prescriptions from two different doctors can combine into an interaction neither of them saw. Watchdog is built to catch that gap: snap a photo of a prescription, pill strip or lab report, and it walks the same reasoning chain a careful pharmacist would.

**prediction → evidence → patient context → explanation → alert**

> ⚠️ Research and portfolio project. Not a medical device and not medical advice. Always confirm with a pharmacist or physician.

## How it works

| Step | Agent | What it does |
|------|-------|--------------|
| 1 | **Vision intake** | Extracts drugs, doses and lab values from a photo (prescription, pill strip, lab report) using a vision LLM |
| 2 | **ML prediction** | Scores interaction severity (0–3) with XGBoost and a Graph Neural Network built on CYP450 enzyme pathways and molecular features; SHAP explains each prediction |
| 3 | **RAG retrieval** | Pulls evidence from DrugBank, DailyMed, FDA FAERS, PubMed and RxNorm with BioBERT embeddings, FAISS + BM25 hybrid search, RxNorm query expansion and cross-encoder reranking |
| 4 | **Patient context** | Adjusts risk for the individual: pregnancy, dialysis, INR, age, conditions, prior alerts |
| 5 | **Explanation** | Writes two reports from the same evidence: a clinical one for prescribers and a plain-language one for patients, both with citations |
| 6 | **Alert routing** | Decides how loud the warning should be and who receives it (patient, pharmacist, physician, EHR), with hard escalation rules for critical cases |

LangGraph orchestrates the agents. An **LLM router** keeps the system up on free tiers by falling back automatically: Groq Llama 3.3 70B → Llama 4 Scout (vision) → DeepSeek R1 via OpenRouter → a local Llama model through Ollama.

## Tech stack

- **ML:** XGBoost, PyTorch Geometric (GNN), SHAP, scikit-learn
- **Retrieval:** BioBERT / sentence-transformers, FAISS, BM25, cross-encoder reranking
- **Agents:** LangGraph, Groq, OpenRouter, Ollama
- **Backend:** FastAPI, Supabase (auth + Postgres)
- **Frontend:** Next.js 15, React 19, Tailwind CSS
- **CI:** GitHub Actions (compile check, pytest, frontend build)

## Project structure

```text
agents/      Orchestrator, LangGraph graph, the six specialist agents, LLM router, memory
api/         FastAPI app, routes (auth, patients, analyse, intake, alerts), schemas
frontend/    Next.js app: auth, onboarding, image upload, analysis dashboard, alerts
ingestion/   DrugBank, DailyMed, FAERS, PubMed, RxNorm parsers and embedding pipeline
ml/          CYP450 + molecular features, XGBoost and GNN training, unified predictor
rag/         Retriever, reranker, query expander, context assembler, citation tracker
tests/       API, alert routing, citation and CYP450 feature tests
```

## Running locally

**1. Configure environment**

```bash
cp .env.example .env            # fill in Supabase and LLM keys
```

**2. Backend**

```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload   # http://localhost:8000/docs
```

**3. Frontend**

```bash
cd frontend
npm install
npm run dev                     # http://localhost:3000
```

**4. Tests**

```bash
pip install pytest httpx
pytest
```

Raw datasets, generated embeddings and secrets are not committed. DrugBank data requires your own academic licence.

## Security notes

- The API only accepts browser requests from origins listed in `ALLOWED_ORIGINS`.
- Auth tokens are verified server-side through Supabase on every protected route.
- Internal errors are logged on the server and never returned to the client.
- Sample images (`labs.png`, `pills.png`) contain no real patient data.

## Authors

Built by **Oindrilla Banerjee** ([@oindrilla007](https://github.com/oindrilla007)) and **Raj Ambavane** ([@RajAmbavane](https://github.com/RajAmbavane)).

Feedback from anyone working in pharmacovigilance, clinical informatics or healthcare AI is very welcome. Open an issue.
