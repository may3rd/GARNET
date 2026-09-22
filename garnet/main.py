#!/usr/bin/env python3
"""Entrypoint for the GARNET detection API: `python garnet/main.py`.

Canonical implementation lives in `garnet/api.py`; this only starts the server, mirroring
`backend/main.py`.

Port 8090 is shared with `backend/api.py` — run one or the other, never both.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run from the repo root so `garnet.api` imports whether this was started from garnet/, the
# repo root, or anywhere else — uvicorn's reloader re-imports by name in a child process that
# inherits cwd but not sys.path.
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import uvicorn

from garnet.api import app  # re-export the canonical FastAPI app  # noqa: E402,F401


if __name__ == "__main__":
    uvicorn.run("garnet.api:app", reload=True, port=8090)
