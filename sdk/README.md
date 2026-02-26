<p align="center">
  <img src="https://raw.githubusercontent.com/junainfinity/Mimir/main/assets/banner.jpg" alt="Mimir — The Well of Knowledge, Rewritten" width="100%">
</p>

# mimir-memory (npm)

> TypeScript/JavaScript SDK for the [Mimir](https://github.com/junainfinity/Mimir) agentic memory system.

## Prerequisites

Mimir's Python server must be running:

```bash
pip install mimir-memory[server]
mimir-server              # starts on http://localhost:8484
```

## Install

```bash
npm install mimir-memory
```

## Usage

```typescript
import { Mimir } from "mimir-memory";

const mimir = new Mimir(); // defaults to http://localhost:8484

// Archive a fact
await mimir.archive({
  content: "John lives in London",
  source: "user",
  relation: "lives_in",
  target: "London",
  scope: "user",
});

// Search memory
const result = await mimir.search({ query: "Where does John live?" });
console.log(result);
// { result: "Found Context:\n- [... → Present]: user --(lives_in)--> London" }

// Temporal query
const past = await mimir.search({
  query: "Where did John live?",
  timestamp: "2026-01-01T00:00:00",
});

// Health check
const health = await mimir.health();
console.log(health); // { status: "ok", version: "0.1.0" }
```

## API

### `new Mimir(options?)`
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `baseUrl` | `string` | `http://localhost:8484` | Mimir server URL |

### `mimir.archive(params)`
Archive a fact into bitemporal memory.

### `mimir.search(params)`
Search memory with optional temporal filtering.

### `mimir.health()`
Check server status.
