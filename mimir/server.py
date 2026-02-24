"""
Mimir REST API Server

Exposes ``archive_memory`` and ``search_memory`` as HTTP endpoints
so that non-Python clients (JS/TS, Go, Rust, etc.) can use Mimir.

Usage::

    pip install mimir-memory[server]
    mimir-server                         # starts on http://localhost:8484
    mimir-server --port 9000 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mimir.storage import MimirStorage
from mimir.tools import archive_memory, create_tools, search_memory

# ── Pydantic Models ──────────────────────────────────────────────────────

class ArchiveRequest(BaseModel):
    content: str = Field(..., description="Raw text to embed and store")
    source: str = Field(..., description="Source entity (e.g. 'user')")
    relation: str = Field(..., description="Relationship type (e.g. 'lives_in')")
    target: str = Field(..., description="Target entity (e.g. 'London')")
    scope: str = Field("user", description="Scope tag: user, session, or system")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    timestamp: str | None = Field(None, description="ISO timestamp for temporal query")


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class MemoryResponse(BaseModel):
    result: str


# ── App Factory ──────────────────────────────────────────────────────────

def create_app(data_dir: str = "./mimir_data") -> FastAPI:
    """Create and configure the FastAPI application."""
    from mimir import __version__

    app = FastAPI(
        title="Mimir Memory API",
        description="Zero-server agentic memory — archive and search with bitemporal guarantees.",
        version=__version__,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize storage and bind tools
    storage = MimirStorage(data_dir=data_dir)
    create_tools(storage)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok", version=__version__)

    @app.post("/archive", response_model=MemoryResponse)
    async def archive(req: ArchiveRequest):
        try:
            result = archive_memory.invoke({
                "content": req.content,
                "source": req.source,
                "relation": req.relation,
                "target": req.target,
                "scope": req.scope,
            })
            return MemoryResponse(result=str(result))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/search", response_model=MemoryResponse)
    async def search(req: SearchRequest):
        try:
            args: dict = {"query": req.query}
            if req.timestamp:
                args["timestamp"] = req.timestamp
            result = search_memory.invoke(args)
            return MemoryResponse(result=str(result))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ── CLI Entry Point ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mimir Memory REST Server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8484, help="Bind port")
    parser.add_argument("--data-dir", default="./mimir_data", help="Data directory")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install mimir-memory[server]")
        sys.exit(1)

    app = create_app(data_dir=args.data_dir)
    print(f"🧠 Mimir server starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
