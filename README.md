# Mimir — Local Agentic Memory System

> A zero-server, in-process memory system for LLM agents, optimized for Apple Silicon.

Mimir combines **dense vector search** (Zvec), **bitemporal knowledge graphs** (SQLite), and **autonomous tool-calling** (LangGraph) into a single Python process. No Docker, no cloud databases, no external servers.

## ✨ Key Features

- **Bitemporal Memory** — Facts are never deleted. Old records are time-capped (`valid_to`), preserving full history. Ask "Where does John live?" *and* "Where did John live last month?"
- **In-Process Vector Search** — Zvec (Alibaba Proxima) runs as a C-extension inside your Python process. Sub-millisecond search, zero network hops.
- **Fully Local** — Embeddings via HuggingFace BGE run on CPU. Storage is file-based. The only external call is to your LLM endpoint.
- **LangGraph Agent** — Tools (`archive_memory`, `search_memory`) are bound to the LLM and routed via a compiled state graph.

## 🏗️ Architecture

| Layer | Component | Role |
|-------|-----------|------|
| L1 & L2 | Tools | `archive_memory` and `search_memory` — the agent's interface to storage |
| L3 | Storage | Zvec (dense vectors) + SQLite (bitemporal graph edges) |
| L4 | Optimizer | Post-session trajectory analysis (stub) |

## 🚀 Quick Start

```bash
# Requires Python 3.10-3.12 (for Zvec binary compatibility)
# Install Python 3.12 via pyenv if needed:
pyenv install 3.12.8
~/.pyenv/versions/3.12.8/bin/python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run
python3 mimir.py
```

## 📦 Stack

| Component | Technology |
|-----------|-----------|
| Vector Storage | Zvec v0.2.0 (Alibaba Proxima) |
| Graph/Relational | SQLite3 (Python stdlib) |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim, local) |
| Agent Framework | LangGraph + LangChain Core |
| LLM | Any OpenAI-compatible endpoint |

## 📄 Documentation

- **[PROSPECTUS.md](./PROSPECTUS.md)** — Full architecture deep-dive, competitive analysis vs Letta/Mem0/Zep/LangMem, and Zvec integration details

## 📜 License

MIT
