"""
ingestion/database.py
────────────────────────────────────────────────────────────────────────────────
PostgreSQL schema definition and database connection manager for:
  - Patient profiles        (demographics, medical history)
  - Patient drug lists      (current medications per patient)
  - Interaction alerts      (generated alerts with severity + explanation)
  - Alert feedback          (pharmacist/physician acknowledgements)
  - Drug master table       (canonical drug registry from DrugBank + RxNorm)

Why PostgreSQL?
  Structured patient data needs ACID transactions, foreign key integrity,
  and efficient joins — not a vector store. PostgreSQL sits alongside
  FAISS/ChromaDB: relational data lives here, embeddings live there.

Setup:
  1. Install PostgreSQL locally or use Docker:
       docker run --name drugwatchdog-pg \
         -e POSTGRES_PASSWORD=watchdog \
         -e POSTGRES_DB=drugwatchdog \
         -p 5432:5432 -d postgres:15

  2. Set connection string in .env:
       DATABASE_URL=postgresql://postgres:watchdog@localhost:5432/drugwatchdog

  3. Create all tables:
       python -m ingestion.database --create

Install requirements:
  pip install sqlalchemy psycopg2-binary python-dotenv alembic

Usage:
  from ingestion.database import Database, Patient, DrugList, Alert

  db = Database()
  db.connect()
  db.create_tables()

  # Insert a patient
  patient_id = db.insert_patient({
      "name": "John Doe", "age": 72, "weight_kg": 80,
      "egfr": 45.0, "has_liver_disease": False
  })

  # Insert their drug list
  db.insert_drug_list(patient_id, ["warfarin", "aspirin", "amiodarone"])

  # Log an alert
  db.insert_alert({
      "patient_id": patient_id,
      "drug_a": "warfarin", "drug_b": "aspirin",
      "severity": 3, "alert_type": "pharmacist",
      "explanation": "Major bleeding risk ..."
  })
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Default connection string (override via DATABASE_URL env var) ─────────────
DEFAULT_DATABASE_URL = "postgresql://postgres:watchdog@localhost:5432/drugwatchdog"


# ─────────────────────────────────────────────────────────────────────────────
# Schema DDL
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
-- ── Extension ─────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Drug master table ──────────────────────────────────────────────────────
-- Canonical drug registry built from DrugBank + RxNorm mapping.
-- All other tables reference drugs by rxnorm_cui or drugbank_id.
CREATE TABLE IF NOT EXISTS drugs (
    id              SERIAL PRIMARY KEY,
    rxnorm_cui      VARCHAR(20)  UNIQUE,
    drugbank_id     VARCHAR(20) UNIQUE,
    canonical_name  VARCHAR(255) NOT NULL,
    synonyms        TEXT,                    -- pipe-separated list
    drug_type       VARCHAR(50),             -- small-molecule | biotech
    atc_codes       TEXT,                    -- pipe-separated ATC codes
    molecular_weight NUMERIC(10, 3),
    logp             NUMERIC(6, 3),
    half_life        TEXT,
    protein_binding  TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_drugs_rxnorm    ON drugs(rxnorm_cui);
CREATE INDEX IF NOT EXISTS idx_drugs_drugbank  ON drugs(drugbank_id);
CREATE INDEX IF NOT EXISTS idx_drugs_name      ON drugs(canonical_name);

-- ── Patients ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS patients (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    external_id     VARCHAR(100) UNIQUE,     -- hospital MRN or user-assigned ID
    name            VARCHAR(255),
    age             INTEGER,
    sex             VARCHAR(10),             -- M | F | Other
    weight_kg       NUMERIC(6, 2),
    height_cm       NUMERIC(6, 2),
    -- Renal / hepatic function (affects drug metabolism)
    egfr            NUMERIC(6, 2),           -- eGFR mL/min/1.73m²
    creatinine      NUMERIC(6, 3),           -- mg/dL
    alt             NUMERIC(8, 2),           -- liver ALT U/L
    ast             NUMERIC(8, 2),           -- liver AST U/L
    -- Clinical flags
    has_liver_disease   BOOLEAN DEFAULT FALSE,
    has_renal_disease   BOOLEAN DEFAULT FALSE,
    is_pregnant         BOOLEAN DEFAULT FALSE,
    is_elderly          BOOLEAN DEFAULT FALSE,   -- age >= 65
    -- Metadata
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patients_external ON patients(external_id);

-- ── Patient drug lists ─────────────────────────────────────────────────────
-- One row per drug per patient. A patient with 5 drugs has 5 rows.
CREATE TABLE IF NOT EXISTS patient_drugs (
    id              SERIAL PRIMARY KEY,
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    drug_id         INTEGER REFERENCES drugs(id),
    drug_name_raw   VARCHAR(255) NOT NULL,   -- original input name
    rxnorm_cui      VARCHAR(20),             -- resolved CUI (may be NULL)
    canonical_name  VARCHAR(255),
    dose_mg         NUMERIC(10, 3),
    frequency       VARCHAR(100),            -- e.g. "twice daily"
    route           VARCHAR(50),             -- oral | IV | topical
    start_date      DATE,
    is_active       BOOLEAN DEFAULT TRUE,
    added_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patient_drugs_patient ON patient_drugs(patient_id);
CREATE INDEX IF NOT EXISTS idx_patient_drugs_rxnorm  ON patient_drugs(rxnorm_cui);

-- ── Known drug-drug interactions ──────────────────────────────────────────
-- Populated from DrugBank parser output.
-- Used as a lookup table at inference time (fast JOIN, no ML needed for known pairs).
CREATE TABLE IF NOT EXISTS known_interactions (
    id              SERIAL PRIMARY KEY,
    drug_a_cui      VARCHAR(20),
    drug_a_name     VARCHAR(255) NOT NULL,
    drug_b_cui      VARCHAR(20),
    drug_b_name     VARCHAR(255) NOT NULL,
    severity        SMALLINT NOT NULL DEFAULT 0,  -- 0=unknown 1=minor 2=moderate 3=major
    description     TEXT,
    source          VARCHAR(50) DEFAULT 'drugbank',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    -- Canonical pair key for fast lookup (always drug_a < drug_b alphabetically)
    pair_key        VARCHAR(512) GENERATED ALWAYS AS (
                        LEAST(drug_a_name, drug_b_name) || '_' ||
                        GREATEST(drug_a_name, drug_b_name)
                    ) STORED,
    UNIQUE(pair_key)
);

CREATE INDEX IF NOT EXISTS idx_interactions_pair    ON known_interactions(pair_key);
CREATE INDEX IF NOT EXISTS idx_interactions_sev     ON known_interactions(severity);
CREATE INDEX IF NOT EXISTS idx_interactions_drug_a  ON known_interactions(drug_a_name);
CREATE INDEX IF NOT EXISTS idx_interactions_drug_b  ON known_interactions(drug_b_name);

-- ── Alerts ────────────────────────────────────────────────────────────────
-- One row per generated alert per drug pair per patient analysis.
CREATE TABLE IF NOT EXISTS alerts (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    drug_a          VARCHAR(255) NOT NULL,
    drug_b          VARCHAR(255) NOT NULL,
    severity        SMALLINT NOT NULL,       -- 0–3
    severity_label  VARCHAR(20),             -- none | minor | moderate | major
    alert_type      VARCHAR(50),             -- pharmacist | physician | patient | log_only
    -- ML model outputs
    ml_score        NUMERIC(6, 4),           -- GNN/XGB predicted probability
    ml_model        VARCHAR(50),             -- gnn | xgboost
    -- RAG evidence
    evidence_chunks TEXT,                    -- JSON array of chunk_ids used
    citation_urls   TEXT,                    -- pipe-separated source URLs
    -- LLM-generated explanations
    explanation_clinical  TEXT,              -- for pharmacist / physician
    explanation_patient   TEXT,              -- plain language for patient
    -- Contextual adjustments made by patient_context_agent
    adjusted_for_renal    BOOLEAN DEFAULT FALSE,
    adjusted_for_hepatic  BOOLEAN DEFAULT FALSE,
    adjusted_for_age      BOOLEAN DEFAULT FALSE,
    -- Status
    status          VARCHAR(30) DEFAULT 'generated',  -- generated | acknowledged | dismissed
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    -- Metadata
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_patient   ON alerts(patient_id);
CREATE INDEX IF NOT EXISTS idx_alerts_severity  ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_status    ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_created   ON alerts(created_at DESC);

-- ── Alert feedback ────────────────────────────────────────────────────────
-- Tracks pharmacist / physician actions on alerts (for future model retraining).
CREATE TABLE IF NOT EXISTS alert_feedback (
    id              SERIAL PRIMARY KEY,
    alert_id        UUID NOT NULL REFERENCES alerts(id) ON DELETE CASCADE,
    reviewer_role   VARCHAR(50),             -- pharmacist | physician
    action          VARCHAR(50),             -- acknowledged | dismissed | escalated
    comment         TEXT,
    was_correct     BOOLEAN,                 -- did the reviewer agree with the alert?
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_alert ON alert_feedback(alert_id);

-- ── Analysis sessions ─────────────────────────────────────────────────────
-- Tracks each time a patient's drug list is analysed (audit trail).
CREATE TABLE IF NOT EXISTS analysis_sessions (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id      UUID NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    drug_count      INTEGER,
    pair_count      INTEGER,                 -- number of pairs analysed
    alert_count     INTEGER,
    high_sev_count  INTEGER,                 -- severity >= 2
    duration_ms     INTEGER,                 -- total processing time
    agent_trace     TEXT,                    -- JSON log of agent calls
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_patient ON analysis_sessions(patient_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON analysis_sessions(created_at DESC);
"""

# ── Trigger: auto-update patients.updated_at ─────────────────────────────────
TRIGGER_SQL = """
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_patients_updated_at ON patients;
CREATE TRIGGER trg_patients_updated_at
    BEFORE UPDATE ON patients
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
"""


# ─────────────────────────────────────────────────────────────────────────────
# Database manager
# ─────────────────────────────────────────────────────────────────────────────

class Database:
    """
    Thin wrapper around psycopg2 providing:
      - Connection management with context manager
      - Table creation
      - CRUD helpers for all core tables
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        self.database_url = (
            database_url
            or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
        )
        self._conn = None

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> "Database":
        """Open a persistent connection. Call once at app startup."""
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise ImportError("Run: pip install psycopg2-binary")

        self._conn = psycopg2.connect(self.database_url)
        self._conn.autocommit = False
        logger.info(f"✅ Connected to PostgreSQL: {self._safe_url()}")
        return self

    def disconnect(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("Database connection closed")

    @contextmanager
    def transaction(self):
        """Context manager for explicit transactions with auto-rollback on error."""
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception as exc:
            self._conn.rollback()
            logger.error(f"Transaction rolled back: {exc}")
            raise
        finally:
            cur.close()

    # ── Schema management ─────────────────────────────────────────────────────

    def create_tables(self) -> None:
        """Create all tables + triggers if they don't exist."""
        with self.transaction() as cur:
            cur.execute(SCHEMA_SQL)
            cur.execute(TRIGGER_SQL)
        logger.info("✅ All tables created / verified")

    def drop_tables(self, confirm: bool = False) -> None:
        """Drop all tables — use only in dev/test."""
        if not confirm:
            raise RuntimeError("Pass confirm=True to drop all tables")
        tables = [
            "alert_feedback", "analysis_sessions", "alerts",
            "patient_drugs", "patients", "known_interactions", "drugs"
        ]
        with self.transaction() as cur:
            for tbl in tables:
                cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        logger.warning("⚠ All tables dropped")

    # ── Drug master ───────────────────────────────────────────────────────────

    def upsert_drug(self, drug: dict[str, Any]) -> int:
        """
        Insert or update a drug in the master drugs table.
        Returns the drug row id.
        """
        sql = """
            INSERT INTO drugs
                (rxnorm_cui, drugbank_id, canonical_name, synonyms, drug_type,
                 atc_codes, molecular_weight, logp, half_life, protein_binding)
            VALUES
                (%(rxnorm_cui)s, %(drugbank_id)s, %(canonical_name)s, %(synonyms)s,
                 %(drug_type)s, %(atc_codes)s, %(molecular_weight)s, %(logp)s,
                 %(half_life)s, %(protein_binding)s)
            ON CONFLICT (drugbank_id) DO UPDATE SET
                rxnorm_cui      = EXCLUDED.rxnorm_cui,
                canonical_name  = EXCLUDED.canonical_name,
                synonyms        = EXCLUDED.synonyms,
                drug_type       = EXCLUDED.drug_type,
                atc_codes       = EXCLUDED.atc_codes,
                molecular_weight = EXCLUDED.molecular_weight,
                logp            = EXCLUDED.logp,
                half_life       = EXCLUDED.half_life,
                protein_binding = EXCLUDED.protein_binding
            RETURNING id
        """
        with self.transaction() as cur:
            cur.execute(sql, drug)
            return cur.fetchone()[0]

    def bulk_upsert_drugs(self, drugs_df) -> int:
        """Bulk-upsert from a DrugBank DataFrame. Returns count inserted."""
        import pandas as pd

        def clean(value):
            if value is None:
                return None
            try:
                if pd.isna(value):
                    return None
            except (TypeError, ValueError):
                pass
            return value

        count = 0
        for _, row in drugs_df.iterrows():
            try:
                self.upsert_drug({
                    "rxnorm_cui":      clean(row.get("rxnorm_id")),
                    "drugbank_id":     clean(row.get("drugbank_id")),
                    "canonical_name":  clean(row.get("name")) or "",
                    "synonyms":        clean(row.get("synonyms")) or "",
                    "drug_type":       clean(row.get("drug_type")) or "",
                    "atc_codes":       clean(row.get("atc_codes")) or "",
                    "molecular_weight": clean(row.get("molecular_weight")),
                    "logp":            clean(row.get("logp")),
                    "half_life":       clean(row.get("half_life")) or "",
                    "protein_binding": clean(row.get("protein_binding")) or "",
                })
                count += 1
            except Exception as exc:
                logger.debug(f"Skipped drug {row.get('name')}: {exc}")
        logger.info(f"💾 Upserted {count:,} drugs into drug master table")
        return count

    def bulk_insert_interactions(self, interactions_df) -> int:
        """Bulk-insert known interactions from DrugBank DataFrame."""
        sql = """
            INSERT INTO known_interactions
                (drug_a_name, drug_b_name, drug_a_cui, drug_b_cui, severity, description, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (pair_key) DO UPDATE SET
                severity    = GREATEST(known_interactions.severity, EXCLUDED.severity),
                description = EXCLUDED.description
        """
        count = 0
        with self.transaction() as cur:
            for _, row in interactions_df.iterrows():
                try:
                    cur.execute(sql, (
                        str(row.get("drug_a_name", "")),
                        str(row.get("drug_b_name", "")),
                        row.get("drug_a_id"),
                        row.get("drug_b_id"),
                        int(row.get("severity", 0)),
                        str(row.get("description", "")),
                        "drugbank",
                    ))
                    count += 1
                except Exception as exc:
                    logger.debug(f"Skipped interaction: {exc}")
        logger.info(f"💾 Inserted {count:,} known interactions")
        return count

    # ── Patients ──────────────────────────────────────────────────────────────

    def insert_patient(self, patient: dict[str, Any]) -> str:
        """Insert a patient record. Returns the UUID patient_id."""
        sql = """
            INSERT INTO patients
                (external_id, name, age, sex, weight_kg, height_cm,
                 egfr, creatinine, alt, ast,
                 has_liver_disease, has_renal_disease, is_pregnant, is_elderly)
            VALUES
                (%(external_id)s, %(name)s, %(age)s, %(sex)s, %(weight_kg)s, %(height_cm)s,
                 %(egfr)s, %(creatinine)s, %(alt)s, %(ast)s,
                 %(has_liver_disease)s, %(has_renal_disease)s, %(is_pregnant)s, %(is_elderly)s)
            RETURNING id
        """
        defaults = {
            "external_id": None, "name": None, "age": None, "sex": None,
            "weight_kg": None, "height_cm": None, "egfr": None,
            "creatinine": None, "alt": None, "ast": None,
            "has_liver_disease": False, "has_renal_disease": False,
            "is_pregnant": False,
            "is_elderly": (patient.get("age", 0) or 0) >= 65,
        }
        defaults.update(patient)
        with self.transaction() as cur:
            cur.execute(sql, defaults)
            return str(cur.fetchone()[0])

    def get_patient(self, patient_id: str) -> Optional[dict]:
        """Fetch a patient record by UUID."""
        sql = "SELECT * FROM patients WHERE id = %s"
        with self.transaction() as cur:
            cur.execute(sql, (patient_id,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row))

    def update_patient(self, patient_id: str, updates: dict[str, Any]) -> None:
        """Update specific fields on a patient record."""
        if not updates:
            return
        set_clause = ", ".join(f"{k} = %({k})s" for k in updates)
        sql = f"UPDATE patients SET {set_clause} WHERE id = %(patient_id)s"
        updates["patient_id"] = patient_id
        with self.transaction() as cur:
            cur.execute(sql, updates)

    # ── Drug lists ────────────────────────────────────────────────────────────

    def insert_drug_list(
        self,
        patient_id: str,
        drugs: list[str | dict],
    ) -> int:
        """
        Insert a patient's drug list.
        drugs: list of drug name strings OR dicts with {drug_name_raw, dose_mg, …}
        Returns count inserted.
        """
        sql = """
            INSERT INTO patient_drugs
                (patient_id, drug_name_raw, rxnorm_cui, canonical_name,
                 dose_mg, frequency, route)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        rows = []
        for drug in drugs:
            if isinstance(drug, str):
                rows.append((patient_id, drug, None, None, None, None, None))
            else:
                rows.append((
                    patient_id,
                    drug.get("drug_name_raw", ""),
                    drug.get("rxnorm_cui"),
                    drug.get("canonical_name"),
                    drug.get("dose_mg"),
                    drug.get("frequency"),
                    drug.get("route"),
                ))

        with self.transaction() as cur:
            cur.executemany(sql, rows)
        return len(rows)

    def get_patient_drugs(self, patient_id: str) -> list[dict]:
        """Return all active drugs for a patient."""
        sql = """
            SELECT * FROM patient_drugs
            WHERE patient_id = %s AND is_active = TRUE
            ORDER BY added_at
        """
        with self.transaction() as cur:
            cur.execute(sql, (patient_id,))
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def update_drug_rxnorm(self, patient_id: str, drug_name: str, cui: str, canonical: str) -> None:
        """Back-fill RxNorm CUI after mapping."""
        sql = """
            UPDATE patient_drugs
            SET rxnorm_cui = %s, canonical_name = %s
            WHERE patient_id = %s AND drug_name_raw = %s
        """
        with self.transaction() as cur:
            cur.execute(sql, (cui, canonical, patient_id, drug_name))

    # ── Alerts ────────────────────────────────────────────────────────────────

    def insert_alert(self, alert: dict[str, Any]) -> str:
        """Insert a generated alert. Returns UUID alert_id."""
        severity_labels = {0: "none", 1: "minor", 2: "moderate", 3: "major"}
        sql = """
            INSERT INTO alerts (
                patient_id, drug_a, drug_b, severity, severity_label,
                alert_type, ml_score, ml_model,
                evidence_chunks, citation_urls,
                explanation_clinical, explanation_patient,
                adjusted_for_renal, adjusted_for_hepatic, adjusted_for_age
            ) VALUES (
                %(patient_id)s, %(drug_a)s, %(drug_b)s, %(severity)s, %(severity_label)s,
                %(alert_type)s, %(ml_score)s, %(ml_model)s,
                %(evidence_chunks)s, %(citation_urls)s,
                %(explanation_clinical)s, %(explanation_patient)s,
                %(adjusted_for_renal)s, %(adjusted_for_hepatic)s, %(adjusted_for_age)s
            ) RETURNING id
        """
        defaults = {
            "severity_label":      severity_labels.get(alert.get("severity", 0), "unknown"),
            "alert_type":          "log_only",
            "ml_score":            None,
            "ml_model":            None,
            "evidence_chunks":     None,
            "citation_urls":       None,
            "explanation_clinical": None,
            "explanation_patient": None,
            "adjusted_for_renal":  False,
            "adjusted_for_hepatic": False,
            "adjusted_for_age":    False,
        }
        defaults.update(alert)
        with self.transaction() as cur:
            cur.execute(sql, defaults)
            return str(cur.fetchone()[0])

    def get_patient_alerts(
        self,
        patient_id: str,
        min_severity: int = 0,
        status: Optional[str] = None,
    ) -> list[dict]:
        """Fetch alerts for a patient, optionally filtered by severity / status."""
        conditions = ["patient_id = %s", "severity >= %s"]
        params: list[Any] = [patient_id, min_severity]
        if status:
            conditions.append("status = %s")
            params.append(status)
        sql = f"""
            SELECT * FROM alerts
            WHERE {" AND ".join(conditions)}
            ORDER BY severity DESC, created_at DESC
        """
        with self.transaction() as cur:
            cur.execute(sql, params)
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> None:
        sql = """
            UPDATE alerts
            SET status = 'acknowledged',
                acknowledged_by = %s,
                acknowledged_at = NOW()
            WHERE id = %s
        """
        with self.transaction() as cur:
            cur.execute(sql, (acknowledged_by, alert_id))

    def insert_feedback(self, feedback: dict[str, Any]) -> None:
        sql = """
            INSERT INTO alert_feedback (alert_id, reviewer_role, action, comment, was_correct)
            VALUES (%(alert_id)s, %(reviewer_role)s, %(action)s, %(comment)s, %(was_correct)s)
        """
        with self.transaction() as cur:
            cur.execute(sql, feedback)

    # ── Analysis sessions ─────────────────────────────────────────────────────

    def insert_session(self, session: dict[str, Any]) -> str:
        """Log a completed analysis session. Returns session UUID."""
        sql = """
            INSERT INTO analysis_sessions
                (patient_id, drug_count, pair_count, alert_count,
                 high_sev_count, duration_ms, agent_trace)
            VALUES
                (%(patient_id)s, %(drug_count)s, %(pair_count)s, %(alert_count)s,
                 %(high_sev_count)s, %(duration_ms)s, %(agent_trace)s)
            RETURNING id
        """
        defaults = {
            "drug_count": 0, "pair_count": 0, "alert_count": 0,
            "high_sev_count": 0, "duration_ms": None, "agent_trace": None,
        }
        defaults.update(session)
        with self.transaction() as cur:
            cur.execute(sql, defaults)
            return str(cur.fetchone()[0])

    # ── Interaction lookup ────────────────────────────────────────────────────

    def lookup_interaction(self, drug_a: str, drug_b: str) -> Optional[dict]:
        """
        Fast lookup of a known interaction by drug name pair.
        Returns the interaction record or None if not in the known_interactions table.
        """
        pair_key = "_".join(sorted([drug_a.lower(), drug_b.lower()]))
        # Use LOWER() to handle case differences
        sql = """
            SELECT * FROM known_interactions
            WHERE LOWER(LEAST(drug_a_name, drug_b_name) || '_' || GREATEST(drug_a_name, drug_b_name))
                  = %s
            LIMIT 1
        """
        with self.transaction() as cur:
            cur.execute(sql, (pair_key,))
            row = cur.fetchone()
            if row is None:
                return None
            cols = [desc[0] for desc in cur.description]
            return dict(zip(cols, row))

    def get_high_severity_interactions(self, min_severity: int = 2) -> list[dict]:
        """Return all known interactions above a severity threshold."""
        sql = """
            SELECT * FROM known_interactions
            WHERE severity >= %s
            ORDER BY severity DESC
        """
        with self.transaction() as cur:
            cur.execute(sql, (min_severity,))
            cols = [desc[0] for desc in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Returns True if the DB connection is alive."""
        try:
            with self.transaction() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def table_counts(self) -> dict[str, int]:
        """Return row counts for all tables — useful for monitoring."""
        tables = [
            "drugs", "patients", "patient_drugs",
            "known_interactions", "alerts", "alert_feedback", "analysis_sessions"
        ]
        counts = {}
        with self.transaction() as cur:
            for tbl in tables:
                cur.execute(f"SELECT COUNT(*) FROM {tbl}")
                counts[tbl] = cur.fetchone()[0]
        return counts

    def _safe_url(self) -> str:
        """Mask password in the URL for logging."""
        import re
        return re.sub(r":([^@]+)@", ":****@", self.database_url)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Database management for Drug Watchdog")
    ap.add_argument("--create",     action="store_true", help="Create all tables")
    ap.add_argument("--drop",       action="store_true", help="Drop all tables (dev only)")
    ap.add_argument("--counts",     action="store_true", help="Print table row counts")
    ap.add_argument("--seed-drugs", default=None,        help="Path to drugbank_drugs.parquet to seed drug master")
    ap.add_argument("--seed-pairs", default=None,        help="Path to drug_pairs.parquet to seed known interactions")
    ap.add_argument("--db-url",     default=None,        help="Override DATABASE_URL")
    args = ap.parse_args()

    db = Database(database_url=args.db_url)
    db.connect()

    if args.create:
        db.create_tables()

    if args.drop:
        confirm = input("Type 'yes' to drop all tables: ")
        if confirm.strip().lower() == "yes":
            db.drop_tables(confirm=True)

    if args.seed_drugs:
        import pandas as pd
        drugs_df = pd.read_parquet(args.seed_drugs)
        db.bulk_upsert_drugs(drugs_df)

    if args.seed_pairs:
        import pandas as pd
        pairs_df = pd.read_parquet(args.seed_pairs)
        db.bulk_insert_interactions(pairs_df)

    if args.counts:
        counts = db.table_counts()
        print("\n-- Table row counts -----------------------------------")
        for tbl, count in counts.items():
            print(f"  {tbl:<25} {count:>10,}")

    # Always print health check
    status = "healthy" if db.health_check() else "unreachable"
    print(f"\nDatabase status: {status}")
    db.disconnect()
