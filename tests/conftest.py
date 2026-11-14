import sys
from pathlib import Path

# Make the project root importable (agents, api, ml, rag, ingestion).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
