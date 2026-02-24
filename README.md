<p align="center">
  <h1 align="center">Mimir</h1>
  <p align="center"><em>A zero-server, in-process agentic memory system for LLM agents.</em></p>
</p>

[📚 Prospectus](./PROSPECTUS.md) | [🐍 PyPI](https://pypi.org/project/mimir-memory/) | [📦 npm](https://www.npmjs.com/package/mimir-memory) | [🏗️ Architecture](#-architecture) | [⚡ Quickstart](#-one-minute-example)

Mimir is an open-source, in-process memory system for AI agents — combining **dense vector search**, **bitemporal knowledge graphs**, and **autonomous tool-calling** into a single Python process. Built on [Zvec](https://github.com/alibaba/zvec) (Alibaba's battle-tested vector engine) and SQLite, it delivers production-grade semantic memory with temporal reasoning, zero infrastructure, and minimal setup.

## 💫 Features

- **Bitemporal Memory**: Facts are never deleted. Old records are time-capped, preserving full history. Ask *"Where does John live?"* and *"Where did John live last month?"* from the same dataset.
- **In-Process Vector Search**: [Zvec](https://github.com/alibaba/zvec) runs as a C-extension inside your Python process. Sub-millisecond search, zero network hops.
- **Fully Local**: Embeddings via HuggingFace BGE run on CPU. Storage is file-based. No cloud, no Docker, no external databases.
- **Agent-Ready**: LangGraph state graph with `archive_memory` and `search_memory` tools, ready to bind to any OpenAI-compatible LLM.
- **REST API**: Built-in FastAPI server for JS/TS and cross-language access via a single `mimir-server` command.
- **Runs Anywhere**: As an in-process library, Mimir runs wherever your code runs — notebooks, servers, CLI tools, or edge devices.

## 📦 Installation

### [Python](https://pypi.org/project/mimir-memory/)

```
pip install mimir-memory
```

With Zvec (requires Python 3.10–3.12):

```
pip install mimir-memory[zvec]
```

With REST server:

```
pip install mimir-memory[server]
```

### [Node.js](https://www.npmjs.com/package/mimir-memory)

```
npm install mimir-memory
```

> **Note:** The npm package is a TypeScript SDK that connects to the Mimir REST server. Start the server first with `mimir-server`.

### ✅ Supported Platforms

- macOS (ARM64 — Apple Silicon optimized)
- Linux (x86_64, ARM64)

## ⚡ One-Minute Example

### Python (in-process)

```python
from mimir import MimirStorage, create_agent
from langchain_core.messages import HumanMessage

# Initialize storage (creates ./mimir_data with Zvec + SQLite)
storage = MimirStorage()

# Create a LangGraph agent with memory tools bound
graph = create_agent(storage, api_key="sk-...", base_url="https://...")

# Chat — the agent autonomously archives and retrieves memories
result = graph.invoke({"messages": [
    HumanMessage(content="My name is John and I live in Chennai.")
]})
print(result["messages"][-1].content)
```

### Node.js / TypeScript

```typescript
import { Mimir } from "mimir-memory";

const mimir = new Mimir(); // connects to http://localhost:8484

await mimir.archive({
  content: "John lives in London",
  source: "user",
  relation: "lives_in",
  target: "London",
  scope: "user",
});

const result = await mimir.search({ query: "Where does John live?" });
console.log(result);
```

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│  L1 & L2: Tools (archive_memory, search_memory)  │
├──────────┬───────────────────────────────────────┤
│          │  L3: Bitemporal Knowledge Engine       │
│  Zvec    │  ┌──────────────────────────────────┐  │
│  (dense  │  │  SQLite (bitemporal_graph)        │  │
│  vectors)│  │  valid_from / valid_to timestamps │  │
│          │  └──────────────────────────────────┘  │
├──────────┴───────────────────────────────────────┤
│  L4: Procedural Optimizer (trajectory learning)   │
└──────────────────────────────────────────────────┘
```

| Layer | Component | Role |
|-------|-----------|------|
| **L1 & L2** | Tools | `archive_memory` and `search_memory` — the agent's interface to storage |
| **L3** | Storage | Zvec (dense vectors) + SQLite (bitemporal graph edges) |
| **L4** | Optimizer | Post-session trajectory analysis and rule extraction |

## 📊 Stack

| Component | Technology |
|-----------|-----------|
| Vector Storage | [Zvec v0.2.0](https://github.com/alibaba/zvec) (Alibaba Proxima) |
| Graph/Relational | SQLite3 (Python stdlib) |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim, local, CPU) |
| Agent Framework | LangGraph + LangChain Core |
| REST Server | FastAPI + Uvicorn |
| LLM | Any OpenAI-compatible endpoint |

## 🤝 Community

- 📄 **[Full Prospectus](./PROSPECTUS.md)** — Architecture deep-dive, competitive analysis vs Letta/Mem0/Zep/LangMem, and Zvec integration details
- 🐛 **[Issues](https://github.com/junainfinity/Mimir/issues)** — Bug reports and feature requests

## ❤️ Contributing

We welcome contributions! Whether you're fixing a bug, adding a feature, or improving documentation — your help makes Mimir better for everyone.

## 📜 License

[MIT](./LICENSE)
