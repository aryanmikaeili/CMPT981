"""Weights & Biases initialization helper.

The W&B API key is expected in the environment variable ``WANDB_API_KEY``.
If a ``.env`` file is present at the repository root and the variable is
not already set, we will load it from there as a fallback (without
requiring `python-dotenv`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping, Optional

import wandb


def _maybe_load_dotenv() -> None:
    """Populate ``os.environ`` from a project-root ``.env`` file if needed.

    Only sets keys that aren't already in ``os.environ`` so the real
    environment always wins.
    """
    if 'WANDB_API_KEY' in os.environ and os.environ['WANDB_API_KEY']:
        return
    repo_root = Path(__file__).resolve().parents[2]
    dotenv_path = repo_root / '.env'
    if not dotenv_path.is_file():
        return
    with dotenv_path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def init_wandb(project: str,
               run_name: str,
               config: Mapping[str, object],
               tags: Optional[Iterable[str]] = None,
               group: Optional[str] = None):
    """Login + start a W&B run; raises if WANDB_API_KEY isn't set."""
    _maybe_load_dotenv()
    api_key = os.environ.get('WANDB_API_KEY', '').strip()
    if not api_key:
        raise RuntimeError(
            'WANDB_API_KEY is not set. Export it in your shell '
            '(e.g. `export WANDB_API_KEY=...`) or put it in a `.env` '
            'file at the project root before running.'
        )
    wandb.login(key=api_key)
    return wandb.init(
        project=project,
        name=run_name,
        config=dict(config),
        tags=list(tags) if tags else None,
        group=group,
    )
