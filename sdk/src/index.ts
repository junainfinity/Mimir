/**
 * Mimir Memory SDK
 *
 * TypeScript/JavaScript client for the Mimir agentic memory REST API.
 *
 * @example
 * ```ts
 * import { Mimir } from "mimir-memory";
 *
 * const mimir = new Mimir();  // defaults to http://localhost:8484
 *
 * await mimir.archive({
 *   content: "John lives in London",
 *   source: "user",
 *   relation: "lives_in",
 *   target: "London",
 *   scope: "user",
 * });
 *
 * const results = await mimir.search({ query: "Where does John live?" });
 * console.log(results);
 * ```
 */

// ── Types ───────────────────────────────────────────────────────────────

export interface MimirOptions {
    /** Base URL of the Mimir REST server. Defaults to `http://localhost:8484`. */
    baseUrl?: string;
}

export interface ArchiveParams {
    /** Raw text content to embed and store. */
    content: string;
    /** Source entity (e.g. "user"). */
    source: string;
    /** Relationship type (e.g. "lives_in"). */
    relation: string;
    /** Target entity (e.g. "London"). */
    target: string;
    /** Scope tag: "user", "session", or "system". Defaults to "user". */
    scope?: string;
}

export interface SearchParams {
    /** Natural language search query. */
    query: string;
    /** Optional ISO timestamp for temporal queries. */
    timestamp?: string;
}

export interface MemoryResponse {
    result: string;
}

export interface HealthResponse {
    status: string;
    version: string;
}

// ── Client ──────────────────────────────────────────────────────────────

export class Mimir {
    private baseUrl: string;

    constructor(options: MimirOptions = {}) {
        this.baseUrl = (options.baseUrl || "http://localhost:8484").replace(
            /\/$/,
            ""
        );
    }

    // -- helpers --

    private async request<T>(
        path: string,
        method: "GET" | "POST",
        body?: unknown
    ): Promise<T> {
        const url = `${this.baseUrl}${path}`;
        const init: RequestInit = {
            method,
            headers: { "Content-Type": "application/json" },
        };
        if (body !== undefined) {
            init.body = JSON.stringify(body);
        }

        const res = await fetch(url, init);

        if (!res.ok) {
            const text = await res.text();
            throw new Error(`Mimir API error ${res.status}: ${text}`);
        }

        return (await res.json()) as T;
    }

    // -- public API --

    /**
     * Check if the Mimir server is reachable.
     */
    async health(): Promise<HealthResponse> {
        return this.request<HealthResponse>("/health", "GET");
    }

    /**
     * Archive a new fact into Mimir's bitemporal memory.
     *
     * The content is embedded via the server's local embedding model,
     * stored in Zvec, and a bitemporal edge is created in SQLite.
     */
    async archive(params: ArchiveParams): Promise<MemoryResponse> {
        return this.request<MemoryResponse>("/archive", "POST", {
            content: params.content,
            source: params.source,
            relation: params.relation,
            target: params.target,
            scope: params.scope ?? "user",
        });
    }

    /**
     * Search Mimir's memory with a natural language query.
     *
     * Optionally pass a `timestamp` to retrieve facts valid at that point in time.
     */
    async search(params: SearchParams): Promise<MemoryResponse> {
        const body: Record<string, string> = { query: params.query };
        if (params.timestamp) {
            body.timestamp = params.timestamp;
        }
        return this.request<MemoryResponse>("/search", "POST", body);
    }
}

export default Mimir;
