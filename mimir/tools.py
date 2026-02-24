"""
L1 & L2: OS Context Manager & Semantic Router (Tool Definitions)

Provides ``archive_memory`` and ``search_memory`` as LangChain tools
that read/write to a shared ``MimirStorage`` instance.
"""

from __future__ import annotations

import datetime
from typing import Optional

from langchain_core.tools import tool

try:
    import zvec
except ImportError:
    from mimir._mock_zvec import install_mock
    zvec = install_mock()

# Module-level storage reference — set by ``create_tools()``
_storage = None


def create_tools(storage):
    """
    Bind tools to a ``MimirStorage`` instance and return them as a list.

    Args:
        storage: A ``MimirStorage`` instance.

    Returns:
        list: ``[archive_memory, search_memory]``
    
    Example::

        from mimir import MimirStorage, create_tools
        storage = MimirStorage()
        tools = create_tools(storage)
    """
    global _storage
    _storage = storage
    return [archive_memory, search_memory]


@tool
def archive_memory(
    content: str, source: str, relation: str, target: str, scope: str
) -> str:
    """Archives a new fact into memory.

    Embeds the content in Zvec and updates the SQLite bitemporal graph.
    If the source-relation already exists, it invalidates the old record.
    """
    print(f"[archive_memory] {source} -> {relation} -> {target} ({scope})")

    # 1. Embed
    try:
        embedding = _storage.embeddings.embed_query(content)
    except Exception as e:
        print(f"Warning: embedding failed ({e}), using zeros.")
        embedding = [0.0] * _storage.embedding_dim

    zvec_id = f"chunk_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"

    # 2. Store in Zvec
    try:
        doc = zvec.Doc(
            id=zvec_id,
            vectors={"embedding": embedding},
            fields={"text": content},
        )
        _storage.collection.insert([doc])
    except Exception as e:
        print(f"Zvec insert note: {e}")

    # 3. Update SQLite
    conn = _storage.get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE bitemporal_graph SET valid_to = CURRENT_TIMESTAMP "
        "WHERE source_entity = ? AND relationship = ? AND valid_to IS NULL",
        (source, relation),
    )
    cur.execute(
        "INSERT INTO bitemporal_graph "
        "(source_entity, relationship, target_entity, scope_tag, zvec_reference_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (source, relation, target, scope, zvec_id),
    )
    conn.commit()
    conn.close()
    return f"Archived: {source} --({relation})--> {target}"


@tool
def search_memory(query: str, timestamp: Optional[str] = None) -> str:
    """Searches the memory graph based on a semantic query.

    If a timestamp is provided, returns information valid at that time.
    Otherwise returns currently valid relationships.
    """
    print(f"[search_memory] query='{query}' time={timestamp or 'CURRENT'}")

    # 1. Embed
    try:
        embedding = _storage.embeddings.embed_query(query)
    except Exception:
        embedding = [0.0] * _storage.embedding_dim

    # 2. Zvec search
    try:
        results = _storage.collection.query(
            vectors=zvec.VectorQuery(field_name="embedding", vector=embedding),
            topk=5,
        )
        if not results:
            return "No semantic matches found."
        zvec_ids = [doc.id for doc in results]
    except Exception as e:
        print(f"Zvec search note: {e}")
        zvec_ids = []

    if not zvec_ids:
        return "No relevant memories found."

    # 3. Cross-reference SQLite
    conn = _storage.get_connection()
    cur = conn.cursor()
    placeholders = ",".join("?" * len(zvec_ids))
    params: list = list(zvec_ids)

    sql = (
        "SELECT source_entity, relationship, target_entity, valid_from, valid_to "
        f"FROM bitemporal_graph WHERE zvec_reference_id IN ({placeholders})"
    )
    if timestamp:
        sql += " AND valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?)"
        params.extend([timestamp, timestamp])
    else:
        sql += " AND valid_to IS NULL"

    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return "Matching text found, but not valid at the requested time."

    lines = ["Found Context:"]
    for src, rel, tgt, vf, vt in rows:
        lines.append(f"- [{vf} → {vt or 'Present'}]: {src} --({rel})--> {tgt}")
    return "\n".join(lines)
