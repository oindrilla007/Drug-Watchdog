# Drug Interaction Watchdog

Drug Interaction Watchdog is an AI-assisted pharmacovigilance project for detecting and explaining potentially dangerous drug-drug interactions. The system combines structured clinical features, machine learning, graph neural networks, retrieval-augmented generation, and multi-agent orchestration to support safer medication review workflows.

## Current Capabilities

- Drug interaction prediction with XGBoost and GNN model paths.
- Feature engineering for CYP450 signals, molecular properties, and label-derived warnings.
- FastAPI backend scaffolding for patient, analysis, and alert routes.
- RAG components for retrieval, reranking, query expansion, context assembly, and citation tracking.
- Multi-agent modules for orchestration, ML prediction, patient context, explanation, retrieval, memory, and alert routing.
- Evaluation scripts for model comparison, SHAP analysis, and GNN artifact loading.

## Project Structure

```text
agents/      Multi-agent orchestration and specialist agents
api/         FastAPI app, routes, schemas, and database setup
configs/     Model, RAG, and agent configuration
ingestion/   DrugBank, DailyMed, FAERS, PubMed, RxNorm, and embedding pipelines
ml/          Features, models, training, evaluation, and prediction interface
rag/         Retrieval, reranking, context assembly, citations, and expansion
tests/       API, ingestion, ML, RAG, and agent tests
```

## Quick Start

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m ml.predictor
```

Large raw datasets, generated embeddings, processed data, and local secrets are intentionally excluded from Git.
