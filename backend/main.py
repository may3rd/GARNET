"""Compatibility entrypoint for GARNET API.

Canonical backend implementation lives in `api.py`.
This module exists to keep older commands (`uvicorn main:app`) working
without maintaining a second diverging backend implementation.
"""

import uvicorn

from api import app  # re-export canonical FastAPI app


if __name__ == "__main__":
    uvicorn.run("api:app", reload=True, port=8001)
