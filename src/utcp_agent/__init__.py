"""UtcpAgent: A LangGraph-based agent for UTCP tool calling.

This package provides a ready-to-use agent implementation that uses LangGraph
to orchestrate UTCP tool discovery and execution.
"""

from utcp_agent.utcp_agent import UtcpAgent, UtcpAgentConfig, AgentState

__version__ = "1.0.0"
__all__ = ["UtcpAgent", "UtcpAgentConfig", "AgentState"]
