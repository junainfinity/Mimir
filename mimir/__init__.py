"""
Mimir — Zero-server, in-process agentic memory system for LLM agents.

Combines dense vector search (Zvec), bitemporal knowledge graphs (SQLite),
and autonomous tool-calling (LangGraph) into a single Python process.
"""

from mimir.storage import MimirStorage
from mimir.tools import create_tools
from mimir.graph import create_agent
from mimir.optimizer import optimize_trajectory

__version__ = "0.3.1"
__all__ = ["MimirStorage", "create_tools", "create_agent", "optimize_trajectory"]
