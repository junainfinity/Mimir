"""
Mimir Demo — Demonstrates bitemporal memory in action.

Usage::

    # Requires Python 3.10-3.12 for real Zvec, works with mock on 3.13+
    python examples/demo.py

Set your LLM endpoint via environment variables or edit the constants below.
"""

import asyncio

from langchain_core.messages import HumanMessage

from mimir import MimirStorage, create_agent, optimize_trajectory


# ── Configuration ────────────────────────────────────────────────────────
LLM_API_KEY = "osm_eAq4kL1l2JoNbAHCtoArK2LrWUoLkK9LiFLRsk5E"
LLM_BASE_URL = "https://api.osmAPI.com/v1"
LLM_MODEL = "gpt-4o-mini"
# ─────────────────────────────────────────────────────────────────────────


async def main():
    print("Initializing Mimir Memory System...\n")

    storage = MimirStorage()
    graph = create_agent(
        storage,
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )

    prompts = [
        "Hi Mimir! My name is John and I live in Chennai.",
        "Can you verify where I live?",
        "I just moved to London. Can you update my profile?",
        "Where do I live currently?",
        "Where did I live before?",
    ]

    session: list = []
    for prompt in prompts:
        print(f"\nUser: {prompt}")
        session.append(HumanMessage(content=prompt))
        try:
            state = graph.invoke({"messages": session})
            reply = state["messages"][-1]
            print(f"Mimir: {reply.content}")
            session = state["messages"]
        except Exception as e:
            print(f"[Error] {e}")
            break

    # Run optimizer stub
    await optimize_trajectory(session)

    # Inspect SQLite
    print("\n--- Inspecting SQLite State ---")
    conn = storage.get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT source_entity, relationship, target_entity, "
        "valid_from, valid_to FROM bitemporal_graph"
    )
    for row in cur.fetchall():
        print(f"  {row[0]} --({row[1]})--> {row[2]} | {row[3]} → {row[4] or 'NOW'}")
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
