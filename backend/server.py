"""
FastAPI server with Server-Sent Events streaming for the peptide agent.
"""

from __future__ import annotations
import json
import logging
import os
import sys
import asyncio
from pathlib import Path
from typing import AsyncIterator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Allow running as `python server.py` from the backend dir
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field

from backend.agent import PepForgeAgent, AgentResult, AttemptLog
from backend import models

app = FastAPI(title="PepForge API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    # Range-based inputs (preferred — sent by the new frontend form)
    length_min: int | None = Field(None, ge=2, le=200)
    length_max: int | None = Field(None, ge=2, le=200)
    charge_min: float | None = None
    charge_max: float | None = None
    hydro_min: float | None = None   # hydrophobic % 0-100
    hydro_max: float | None = None   # hydrophobic % 0-100
    # Legacy single-value fallbacks (still accepted for eval scripts)
    length: int = Field(12, ge=2, le=200)
    charge: float = 0.0
    hydrophobicity: float = 0.0
    activities: list[str] = Field(default_factory=list)
    mol_weight: float | None = None
    hydro_percent: float | None = None
    aromaticity_percent: float | None = None
    reference: str | None = None
    max_retries: int = Field(6, ge=1, le=10)
    threshold: float = Field(0.35, ge=0.0, le=1.0)
    prompt_override: str | None = None

# ── SSE streaming endpoint ───────────────────────────────────────────────────

async def _generate_stream(req: GenerateRequest) -> AsyncIterator[dict]:
    task = req.model_dump()
    if req.reference is None:
        task['reference'] = ''

    # When range params are provided, derive single-value equivalents used by
    # reference selection and the assistant primer logic in agent.py.
    if req.length_min is not None and req.length_max is not None:
        task['length'] = req.length_max          # size max_tokens conservatively
    if req.charge_min is not None and req.charge_max is not None:
        task['charge'] = (req.charge_min + req.charge_max) / 2
    if req.hydro_min is not None and req.hydro_max is not None:
        task['ref_hydrophobic_pct'] = int((req.hydro_min + req.hydro_max) / 2)
        task.pop('hydrophobicity', None)

    loop = asyncio.get_event_loop()
    attempt_queue: asyncio.Queue = asyncio.Queue()

    def on_attempt(log: AttemptLog):
        status = "pass" if log.passed else "fail" if log.sequence else "error"
        from .rulebook import rulebook_score as _rb_score
        rb = _rb_score(log.rulebook) if log.rulebook else None
        loop.call_soon_threadsafe(
            attempt_queue.put_nowait,
            {
                "type": "attempt",
                "n": log.n,
                "sequence": log.sequence,
                "score": log.score,
                "rulebook_score": rb,
                "status": status,
                "issues": log.issues,
                "passed": log.passed,
                "components": log.components,
                "rulebook": log.rulebook,
            }
        )

    agent = PepForgeAgent(
        threshold=req.threshold,
        max_retries=req.max_retries,
    )

    # Run agent in thread pool so we don't block the event loop
    result_holder: list[AgentResult] = []

    def run_agent():
        result = agent.generate(task, on_attempt=on_attempt)
        result_holder.append(result)
        loop.call_soon_threadsafe(attempt_queue.put_nowait, {"type": "__done__"})

    import concurrent.futures
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, run_agent)

    # Stream attempt events as they arrive
    while True:
        event = await attempt_queue.get()
        if event.get("type") == "__done__":
            break
        yield {"data": json.dumps(event)}

    # Final result
    if result_holder:
        r = result_holder[0]
        final = {
            "type": "final",
            "result": {
                "sequence": r.sequence,
                "score": r.score,
                "rulebook_score": r.rulebook_score,
                "reference_used": r.reference_used,
                "components": r.components,
                "rulebook": r.rulebook,
                "iterations": r.iterations,
                "trace": r.trace,
                "time_seconds": r.time_seconds,
            }
        }

        # pLDDT scoring — additive only, computed once on the final sequence,
        # never affects generation/retry logic. Never raises. Run off the
        # event loop since get_plddt() does a blocking HTTP call (up to ~60s).
        try:
            from .esmfold_scorer import get_plddt
            if r.sequence:
                plddt = await loop.run_in_executor(executor, get_plddt, r.sequence)
                final["result"]["plddt_score"]      = plddt.get("mean_plddt")
                final["result"]["plddt_confidence"] = plddt.get("confidence")
                final["result"]["plddt_passes"]     = plddt.get("passes", False)
                final["result"]["plddt_interp"]     = plddt.get("interpretation")
                final["result"]["plddt_pdb"]        = plddt.get("pdb")
            else:
                final["result"]["plddt_score"] = final["result"]["plddt_confidence"] = None
                final["result"]["plddt_passes"] = False
                final["result"]["plddt_interp"] = "No sequence generated"
                final["result"]["plddt_pdb"] = None
        except Exception:
            final["result"]["plddt_score"] = final["result"]["plddt_confidence"] = None
            final["result"]["plddt_passes"] = False
            final["result"]["plddt_interp"] = "pLDDT scoring unavailable"
            final["result"]["plddt_pdb"] = None

        yield {"data": json.dumps(final)}


@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    return EventSourceResponse(_generate_stream(req))


# ── Health endpoint ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return models.health_check()


# ── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(Path(__file__).parent.parent),
    )
