"""
ingestion/pipeline.py
────────────────────────────────────────────────────────────────────────────────
Master orchestration script for Phase 1 — Data Ingestion & Knowledge Base.

Runs every ingestion component in the correct order:

  Step 1  →  Parse DrugBank XML
  Step 2  →  Parse FDA DailyMed SPL labels
  Step 3  →  Load FDA FAERS quarterly CSVs
  Step 4  →  Fetch PubMed abstracts for high-severity drug pairs
  Step 5  →  Map all drug names to RxNorm CUIs
  Step 6  →  Chunk all text uniformly
  Step 7  →  Embed chunks with BioBERT → FAISS + ChromaDB
  Step 8  →  Seed PostgreSQL with drug master + known interactions

Each step is independently resumable — if Step 4 crashes, re-run with
--start-step 4 and it picks up from there using cached/saved outputs.

Usage:
  # Full pipeline (first time)
  python -m ingestion.pipeline

  # Resume from a specific step
  python -m ingestion.pipeline --start-step 4

  # Dev mode — small data, fast iteration
  python -m ingestion.pipeline --dev

  # Skip embedding (no GPU, do it later)
  python -m ingestion.pipeline --skip-embed

  # Skip database seeding (no Postgres yet)
  python -m ingestion.pipeline --skip-db

Run time estimates (full data, CPU only):
  Step 1  DrugBank parse       ~10 min  (1 GB XML)
  Step 2  DailyMed parse       ~45 min  (15K XML files)
  Step 3  FAERS load           ~5  min  (2 quarters)
  Step 4  PubMed fetch         ~20 min  (300 pairs × API calls)
  Step 5  RxNorm mapping       ~30 min  (10K+ names, cached after first run)
  Step 6  Text chunking        ~5  min
  Step 7  BioBERT embedding    ~2  hrs  (100K+ chunks on CPU)
  Step 8  DB seeding           ~5  min
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="a"),
    ]
)
logger = logging.getLogger(__name__)

# ── Default paths ─────────────────────────────────────────────────────────────
PATHS = {
    # Raw inputs
    "drugbank_xml":      "data/raw/drugbank/drugbank_all_full_database.xml",
    "dailymed_dir":      "data/raw/dailymed/",
    "faers_dir":         "data/raw/faers/",
    "pubmed_raw_dir":    "data/raw/pubmed/",
    # Processed outputs
    "processed_dir":     "data/processed/",
    "chunks_dir":        "data/processed/chunks/",
    # Embeddings
    "embeddings_dir":    "data/embeddings/",
    "chroma_dir":        "data/embeddings/chroma_db/",
    # Cache
    "rxnorm_cache":      "data/processed/rxnorm_mapping.json",
}

# ── Dev-mode limits ───────────────────────────────────────────────────────────
DEV_LIMITS = {
    "dailymed_max_files": 500,
    "pubmed_max_pairs":   20,
    "embed_priority_only": True,
}


# ─────────────────────────────────────────────────────────────────────────────
# Step result tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepResult:
    step:       int
    name:       str
    success:    bool
    duration_s: float
    output:     dict     = field(default_factory=dict)
    error:      str      = ""


class PipelineRun:
    """Tracks results of all steps in a single pipeline run."""

    def __init__(self) -> None:
        self.start_time = datetime.now(timezone.utc)
        self.results: list[StepResult] = []

    def record(self, result: StepResult) -> None:
        self.results.append(result)
        status = "OK" if result.success else "FAIL"
        logger.info(
            f"{status} Step {result.step} [{result.name}] "
            f"completed in {result.duration_s:.1f}s"
        )
        if not result.success:
            logger.error(f"   Error: {result.error}")

    def summary(self) -> None:
        total = time.time()
        passed = sum(1 for r in self.results if r.success)
        failed = sum(1 for r in self.results if not r.success)
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()

        print("\n" + "=" * 60)
        print("  PHASE 1 PIPELINE SUMMARY")
        print("=" * 60)
        for r in self.results:
            icon = "OK" if r.success else "FAIL"
            print(f"  {icon} Step {r.step:>2}  {r.name:<30}  {r.duration_s:>6.1f}s")
            for k, v in r.output.items():
                print(f"          {k}: {v}")
        print("-" * 60)
        print(f"  Steps passed: {passed}/{len(self.results)}")
        print(f"  Total time:   {elapsed/60:.1f} min")
        print("=" * 60 + "\n")

    def save_log(self, path: str = "pipeline_run.json") -> None:
        payload = {
            "start_time": self.start_time.isoformat(),
            "steps": [
                {
                    "step":       r.step,
                    "name":       r.name,
                    "success":    r.success,
                    "duration_s": round(r.duration_s, 2),
                    "output":     r.output,
                    "error":      r.error,
                }
                for r in self.results
            ]
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Pipeline log saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline steps
# ─────────────────────────────────────────────────────────────────────────────

def run_step(fn):
    """Decorator: wraps a step function with timing + error handling."""
    def wrapper(run: PipelineRun, cfg: argparse.Namespace) -> bool:
        step_num  = fn.__step__
        step_name = fn.__name__.replace("step_", "").replace("_", " ").title()
        logger.info(f"\n{'-'*60}")
        logger.info(f"  STEP {step_num}: {step_name.upper()}")
        logger.info(f"{'-'*60}")
        t0 = time.time()
        try:
            output = fn(cfg) or {}
            run.record(StepResult(
                step=step_num, name=step_name,
                success=True, duration_s=time.time()-t0, output=output
            ))
            return True
        except Exception as exc:
            logger.exception(f"Step {step_num} failed: {exc}")
            run.record(StepResult(
                step=step_num, name=step_name,
                success=False, duration_s=time.time()-t0, error=str(exc)
            ))
            return False
    wrapper.__step__ = fn.__step__
    wrapper.__name__ = fn.__name__
    return wrapper


# ── Step 1: DrugBank ──────────────────────────────────────────────────────────

@run_step
def step_1_drugbank(cfg: argparse.Namespace) -> dict:
    xml_path = cfg.drugbank_xml
    if not Path(xml_path).exists():
        logger.warning(
            f"DrugBank XML not found at {xml_path}. "
            "Download from https://go.drugbank.com/releases/latest\n"
            "Skipping step — downstream steps may fail."
        )
        return {"status": "skipped — file not found"}

    from ingestion.drugbank_parser import DrugBankParser
    parser = DrugBankParser(xml_path)
    parser.parse()
    parser.save_all(cfg.processed_dir)

    drugs_df = parser.get_drugs_df()
    ix_df    = parser.get_interactions_df()
    cyp_df   = parser.get_cyp450_df()
    return {
        "drugs":        f"{len(drugs_df):,}",
        "interactions": f"{len(ix_df):,}",
        "cyp450":       f"{len(cyp_df):,}",
    }

step_1_drugbank.__step__ = 1


# ── Step 2: DailyMed ──────────────────────────────────────────────────────────

@run_step
def step_2_dailymed(cfg: argparse.Namespace) -> dict:
    dailymed_dir = cfg.dailymed_dir
    if not Path(dailymed_dir).exists() or not any(Path(dailymed_dir).rglob("*.xml")):
        logger.warning(
            f"No DailyMed XML files found under {dailymed_dir}. "
            "Download from https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm\n"
            "Skipping step."
        )
        return {"status": "skipped — no XML files found"}

    from ingestion.dailymed_parser import DailyMedParser
    max_files = DEV_LIMITS["dailymed_max_files"] if cfg.dev else None
    parser = DailyMedParser(dailymed_dir)
    parser.parse(max_files=max_files)
    parser.save_all(cfg.processed_dir)

    labels_df = parser.get_labels_df()
    chunks_df = parser.get_chunks_df()
    return {
        "labels":          f"{len(labels_df):,}",
        "chunks":          f"{len(chunks_df):,}",
        "priority_chunks": f"{chunks_df['priority'].sum():,}",
    }

step_2_dailymed.__step__ = 2


# ── Step 3: FAERS ─────────────────────────────────────────────────────────────

@run_step
def step_3_faers(cfg: argparse.Namespace) -> dict:
    faers_dir = cfg.faers_dir
    if not Path(faers_dir).exists():
        logger.warning(
            f"FAERS directory not found: {faers_dir}. "
            "Download quarterly files from https://fis.fda.gov/extensions/FPD-QDE-FAERS/\n"
            "Skipping step."
        )
        return {"status": "skipped — directory not found"}

    from ingestion.faers_loader import FAERSLoader
    loader = FAERSLoader(faers_dir)
    loader.load()
    loader.save_all(cfg.processed_dir)

    reports_df = loader.get_reports_df()
    co_occur   = loader.get_drug_co_occurrence()
    return {
        "reports":       f"{len(reports_df):,}",
        "co_occurrence": f"{len(co_occur):,}",
    }

step_3_faers.__step__ = 3


# ── Step 4: PubMed ────────────────────────────────────────────────────────────

@run_step
def step_4_pubmed(cfg: argparse.Namespace) -> dict:
    from ingestion.pubmed_fetcher import PubMedFetcher

    fetcher = PubMedFetcher(
        api_key = cfg.ncbi_api_key or os.getenv("NCBI_API_KEY", ""),
        raw_dir = cfg.pubmed_raw_dir,
    )

    # Load high-severity pairs from DrugBank output
    pairs_path = Path(cfg.processed_dir) / "drug_pairs.parquet"
    max_pairs  = DEV_LIMITS["pubmed_max_pairs"] if cfg.dev else cfg.pubmed_max_pairs

    if pairs_path.exists():
        import pandas as pd
        pairs_df  = pd.read_parquet(pairs_path)
        high_sev  = pairs_df[pairs_df["severity"] >= 2]
        pair_list = list(zip(high_sev["drug_a_name"], high_sev["drug_b_name"]))
        logger.info(f"Loaded {len(pair_list):,} high-severity pairs for PubMed queries")
        fetcher.fetch_for_drug_pairs(pair_list, max_pairs=max_pairs)
    else:
        logger.warning("drug_pairs.parquet not found — fetching CYP450 literature only")

    fetcher.fetch_cyp450_literature()
    fetcher.save_all(cfg.processed_dir)

    abstracts_df = fetcher.get_abstracts_df()
    chunks_df    = fetcher.get_chunks_df()
    return {
        "abstracts": f"{len(abstracts_df):,}",
        "chunks":    f"{len(chunks_df):,}",
    }

step_4_pubmed.__step__ = 4


# ── Step 5: RxNorm mapping ────────────────────────────────────────────────────

@run_step
def step_5_rxnorm(cfg: argparse.Namespace) -> dict:
    from ingestion.rxnorm_mapper import RxNormMapper

    mapper = RxNormMapper(cache_path=cfg.rxnorm_cache)
    mapper.map_from_parquets(
        drugbank_path  = Path(cfg.processed_dir) / "drugbank_drugs.parquet",
        faers_path     = Path(cfg.processed_dir) / "faers_drug_events.parquet",
        dailymed_path  = Path(cfg.processed_dir) / "dailymed_labels.parquet",
    )
    mapper.save(cfg.processed_dir)

    df = mapper.get_mapping_df()
    resolved = df["resolved"].sum()
    return {
        "total_names": f"{len(df):,}",
        "resolved":    f"{resolved:,}",
        "coverage":    f"{resolved/max(len(df),1)*100:.1f}%",
    }

step_5_rxnorm.__step__ = 5


# ── Step 6: Text chunking ─────────────────────────────────────────────────────

@run_step
def step_6_chunking(cfg: argparse.Namespace) -> dict:
    from ingestion.text_chunker import TextChunker

    processed = Path(cfg.processed_dir)
    chunks    = Path(cfg.chunks_dir)

    chunker = TextChunker()

    # Add sources — each add_* is tolerant of missing files
    chunker.add_drugbank(
        drugs_path        = processed / "drugbank_drugs.parquet",
        interactions_path = processed / "drug_pairs.parquet",
        cyp450_path       = processed / "cyp450_relationships.parquet",
    )
    chunker.add_dailymed(processed / "dailymed_chunks.parquet")
    chunker.add_faers(processed / "faers_drug_events.parquet")
    chunker.add_pubmed(processed / "chunks" / "pubmed_chunks.parquet")

    chunker.run()
    chunker.save(str(chunks))

    df = chunker.get_chunks_df()
    return {
        "total_chunks":    f"{len(df):,}",
        "priority_chunks": f"{df['priority'].sum():,}",
        "sources":         ", ".join(df["source"].unique().tolist()),
    }

step_6_chunking.__step__ = 6


# ── Step 7: Embedding ─────────────────────────────────────────────────────────

@run_step
def step_7_embedding(cfg: argparse.Namespace) -> dict:
    if cfg.skip_embed:
        logger.info("Skipping embedding (--skip-embed flag set)")
        return {"status": "skipped"}

    from ingestion.embedder import Embedder

    chunks_path    = Path(cfg.chunks_dir) / "all_chunks.parquet"
    priority_only  = cfg.dev or DEV_LIMITS["embed_priority_only"] if cfg.dev else False

    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path} — run step 6 first")

    embedder = Embedder(model_name=cfg.embed_model)
    embedder.load_model()
    embedder.embed_chunks(str(chunks_path), priority_only=priority_only)
    embedder.save_faiss(cfg.embeddings_dir)
    embedder.save_chroma(cfg.chroma_dir)

    # Smoke test
    logger.info("Running smoke test query: 'warfarin aspirin bleeding risk'")
    index, meta_df = embedder.load_faiss(cfg.embeddings_dir)
    results = embedder.query_faiss(
        "warfarin aspirin bleeding risk", index, meta_df, top_k=3
    )
    if not results.empty:
        top = results.iloc[0]
        logger.info(
            f"Top result: [{top['source']} | {top['section_type']} | "
            f"score={top['similarity_score']:.3f}]"
        )

    n_vectors = index.ntotal
    return {
        "vectors_indexed": f"{n_vectors:,}",
        "priority_only":   str(priority_only),
        "smoke_test":      "passed" if not results.empty else "no results",
    }

step_7_embedding.__step__ = 7


# ── Step 8: Database seeding ──────────────────────────────────────────────────

@run_step
def step_8_database(cfg: argparse.Namespace) -> dict:
    if cfg.skip_db:
        logger.info("Skipping database seeding (--skip-db flag set)")
        return {"status": "skipped"}

    from ingestion.database import Database
    import pandas as pd

    db = Database(database_url=cfg.db_url)
    try:
        db.connect()
    except Exception as exc:
        logger.warning(
            f"Could not connect to PostgreSQL: {exc}\n"
            "Start Postgres or use: docker run --name drugwatchdog-pg "
            "-e POSTGRES_PASSWORD=watchdog -e POSTGRES_DB=drugwatchdog "
            "-p 5432:5432 -d postgres:15\n"
            "Skipping step."
        )
        return {"status": f"skipped — {exc}"}

    db.create_tables()

    processed = Path(cfg.processed_dir)
    drug_count = ix_count = 0

    drugs_path = processed / "drugbank_drugs.parquet"
    if drugs_path.exists():
        drugs_df   = pd.read_parquet(drugs_path)
        drug_count = db.bulk_upsert_drugs(drugs_df)

    pairs_path = processed / "drug_pairs.parquet"
    if pairs_path.exists():
        pairs_df = pd.read_parquet(pairs_path)
        ix_count = db.bulk_insert_interactions(pairs_df)

    counts = db.table_counts()
    db.disconnect()

    return {
        "drugs_upserted":        f"{drug_count:,}",
        "interactions_inserted": f"{ix_count:,}",
        "db_table_counts":       str(counts),
    }

step_8_database.__step__ = 8


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────────────────────────────────────

ALL_STEPS = [
    step_1_drugbank,
    step_2_dailymed,
    step_3_faers,
    step_4_pubmed,
    step_5_rxnorm,
    step_6_chunking,
    step_7_embedding,
    step_8_database,
]


def main(cfg: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("  DRUG INTERACTION WATCHDOG - PHASE 1 PIPELINE")
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Mode: {'DEV' if cfg.dev else 'FULL'}")
    logger.info("=" * 60)

    run = PipelineRun()

    steps_to_run = [s for s in ALL_STEPS if s.__step__ >= cfg.start_step]
    if cfg.end_step:
        steps_to_run = [s for s in steps_to_run if s.__step__ <= cfg.end_step]

    logger.info(f"Running steps: {[s.__step__ for s in steps_to_run]}")

    for step_fn in steps_to_run:
        success = step_fn(run, cfg)
        if not success and not cfg.ignore_errors:
            logger.error(
                f"Step {step_fn.__step__} failed. "
                "Use --ignore-errors to continue past failures."
            )
            break

    run.summary()
    run.save_log()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Drug Watchdog Phase 1 — Data Ingestion Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Control flow
    ap.add_argument("--start-step",    type=int, default=1,    help="Resume from this step (1–8)")
    ap.add_argument("--end-step",      type=int, default=None, help="Stop after this step")
    ap.add_argument("--ignore-errors", action="store_true",    help="Continue pipeline even if a step fails")
    ap.add_argument("--dev",           action="store_true",    help="Dev mode: smaller data, faster iteration")

    # Skip flags
    ap.add_argument("--skip-embed",    action="store_true",    help="Skip Step 7 (embedding)")
    ap.add_argument("--skip-db",       action="store_true",    help="Skip Step 8 (database seeding)")

    # Data paths (override defaults)
    ap.add_argument("--drugbank-xml",   default=PATHS["drugbank_xml"])
    ap.add_argument("--dailymed-dir",   default=PATHS["dailymed_dir"])
    ap.add_argument("--faers-dir",      default=PATHS["faers_dir"])
    ap.add_argument("--pubmed-raw-dir", default=PATHS["pubmed_raw_dir"])
    ap.add_argument("--processed-dir",  default=PATHS["processed_dir"])
    ap.add_argument("--chunks-dir",     default=PATHS["chunks_dir"])
    ap.add_argument("--embeddings-dir", default=PATHS["embeddings_dir"])
    ap.add_argument("--chroma-dir",     default=PATHS["chroma_dir"])
    ap.add_argument("--rxnorm-cache",   default=PATHS["rxnorm_cache"])

    # API keys + model
    ap.add_argument("--ncbi-api-key",   default="",             help="NCBI API key for PubMed")
    ap.add_argument("--pubmed-max-pairs", type=int, default=300, help="Max drug pairs to query in PubMed")
    ap.add_argument("--embed-model",    default="dmis-lab/biobert-base-cased-v1.2")
    ap.add_argument("--db-url",         default=None,            help="Override DATABASE_URL")

    cfg = ap.parse_args()
    main(cfg)
