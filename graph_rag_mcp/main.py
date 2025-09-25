"""
Standalone Graph RAG MCP Server for Financial Quarterly Analysis

This server provides comprehensive Graph RAG capabilities for financial data analysis,
specifically designed for quarterly earnings reports and financial statements.

Features:
- Financial entity extraction (Metrics, Segments, Ratios, Balance Sheet Items) 
- Neo4j knowledge graph construction with financial schema
- Natural language graph queries with LLM-generated Cypher
- Multi-LLM support (Gemini, GPT, Llama)
- Hybrid vector + graph search capabilities

Dependencies:
- Neo4j running on localhost:7687 (or configured URI)
- API keys for LLM providers

Port: 9008
"""

import os
import sys
import logging
from dotenv import load_dotenv
from fastmcp import FastMCP

# Load environment variables
load_dotenv()

# Local imports
from config.settings import get_config
from tools import register_all_tools

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    config = get_config()
    
    # Validate configuration on startup
    issues = config.validate()
    if issues:
        logger.warning("Configuration issues detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Configuration validated successfully")
    
    # Create MCP server
    server = FastMCP("graph_rag_tool")
    
    # Register all tools
    register_all_tools(server)
    
    logger.info(f"Starting Graph RAG MCP Server")
    logger.info(f"Default model: {config.default_model}")
    logger.info(f"Neo4j URI: {config.neo4j_uri}")
    logger.info(f"Server: {config.host}:{config.port}")
    
    try:
        server.run(transport="streamable-http", host=config.host, port=config.port)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()