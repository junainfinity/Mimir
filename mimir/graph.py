"""
LangGraph Execution Flow

Constructs a compiled ``StateGraph`` that routes between an Agent node
(LLM with tools) and a Tool execution node.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from mimir.storage import MimirStorage
from mimir.tools import archive_memory, create_tools, search_memory

# Default system prompt
SYSTEM_PROMPT = """
<Core_Identity>
You are Mimir, an autonomous memory agent.
You have strict capabilities to archive new facts into a bitemporal knowledge
graph and search past facts.
When a user provides a verifiable statement about themselves, use the
`archive_memory` tool.
When asked a question about the past or present state, use the
`search_memory` tool.
</Core_Identity>
""".strip()


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def create_agent(
    storage: MimirStorage | None = None,
    *,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0,
    system_prompt: str = SYSTEM_PROMPT,
):
    """
    Build and compile the Mimir LangGraph agent.

    Args:
        storage: A ``MimirStorage`` instance. Created automatically if ``None``.
        model: LLM model name. Defaults to ``gpt-4o-mini``.
        api_key: API key for the LLM endpoint.
        base_url: Base URL for the LLM endpoint (for OpenAI-compatible APIs).
        temperature: LLM temperature. Defaults to 0.
        system_prompt: System prompt injected into every turn.

    Returns:
        A compiled LangGraph ``StateGraph``.

    Example::

        from mimir import MimirStorage, create_agent
        storage = MimirStorage()
        graph = create_agent(storage, api_key="sk-...", base_url="https://...")
        result = graph.invoke({"messages": [HumanMessage(content="Hi!")]})
    """
    if storage is None:
        storage = MimirStorage()

    tools = create_tools(storage)

    llm_kwargs: dict = {"model": model, "temperature": temperature}
    if api_key:
        llm_kwargs["api_key"] = api_key
    if base_url:
        llm_kwargs["base_url"] = base_url
    llm = ChatOpenAI(**llm_kwargs)
    llm_with_tools = llm.bind_tools(tools)

    _system = SystemMessage(content=system_prompt)

    def agent_node(state: State):
        response = llm_with_tools.invoke([_system] + state["messages"])
        return {"messages": [response]}

    def tool_node(state: State):
        last = state["messages"][-1]
        results = []
        if isinstance(last, AIMessage) and last.tool_calls:
            for tc in last.tool_calls:
                fn = {"archive_memory": archive_memory, "search_memory": search_memory}.get(tc["name"])
                if fn:
                    out = fn.invoke(tc["args"])
                    results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
        return {"messages": results}

    def should_continue(state: State):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.set_entry_point("agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()
