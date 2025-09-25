"""
Tools package for Graph RAG MCP Server
Centralized tool registration
"""

from .health_tools import register_health_tools
from .extraction_tools import register_extraction_tools
from .graph_tools import register_graph_tools
from .query_tools import register_query_tools

def register_all_tools(server):
    """Register all tool categories with the server"""
    register_health_tools(server)
    register_extraction_tools(server)
    register_graph_tools(server)
    register_query_tools(server)

__all__ = [
    "register_all_tools",
    "register_health_tools",
    "register_extraction_tools", 
    "register_graph_tools",
    "register_query_tools"
]