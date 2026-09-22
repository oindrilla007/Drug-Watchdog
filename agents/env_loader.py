"""
Shared environment loader for standalone agent scripts.

Agent files are often run from ``D:\drug-interaction-watchdog\agents`` during
local demos, so python-dotenv's default current-directory search can miss the
project-root ``.env`` file. This helper anchors loading to the repository root.
"""

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def load_project_env() -> None:
    """Load the project-root .env without overriding real environment vars."""
    load_dotenv(ENV_PATH)
