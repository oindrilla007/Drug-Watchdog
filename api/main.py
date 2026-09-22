import sys
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Make the project root importable so agents/ can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env before anything else touches os.getenv
try:
    from agents.env_loader import load_project_env
    load_project_env()
except Exception:
    pass

from api.routes import auth, patients, analyze, alerts, intake

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.getLogger(__name__).info("Drug Watchdog API starting up")
    yield
    logging.getLogger(__name__).info("Drug Watchdog API shutting down")


app = FastAPI(
    title="Drug Interaction Watchdog API",
    description=(
        "AI-powered drug interaction detection. "
        "Combines XGBoost ML, RAG retrieval, and multi-agent LLM reasoning "
        "to flag clinically significant drug interactions with patient-specific context."
    ),
    version="5.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,     prefix="/auth",     tags=["auth"])
app.include_router(patients.router, prefix="/patients", tags=["patients"])
app.include_router(analyze.router,  prefix="",          tags=["analyse"])
app.include_router(intake.router,   prefix="/intake",   tags=["intake"])
app.include_router(alerts.router,   prefix="/alerts",   tags=["alerts"])


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "version": "5.0.0"}


@app.get("/", tags=["health"])
async def root():
    return {
        "name": "Drug Interaction Watchdog API",
        "version": "5.0.0",
        "docs": "/docs",
    }
