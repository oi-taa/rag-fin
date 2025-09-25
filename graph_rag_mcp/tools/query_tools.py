 
"""
Natural language query tools for Graph RAG MCP Server
Preserving all original query functionality with fallback patterns
"""

import logging
import asyncio
import concurrent.futures
from fastmcp import FastMCP
from config.settings import get_config
from services.graph_service import GraphBuilder
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)

def register_query_tools(server: FastMCP):
    """Register natural language query tools"""
    
    @server.tool("query_financial_graph", description="Natural language queries on financial knowledge graph")
    def mcp_query_graph(question: str, limit: int = 10):
        """Query graph using natural language"""
        try:
            config = get_config()
            api_key = config.get_api_key_for_model(config.default_model)
            
            builder = GraphBuilder(
                config.neo4j_uri,
                config.neo4j_user, 
                config.neo4j_password,
                api_key,
                config.default_model
            )
            
            async def query_async():
                return await builder.query(question, limit)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, query_async())
                results = future.result(timeout=config.graph_query_timeout)
            
            builder.neo4j.close()
            
            return {
                "status": "success",
                "message": "Graph query executed successfully",
                "question": question,
                "results": results,
                "result_count": len(results)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Graph query failed: {str(e)}",
                "error_details": str(e),
                "question": question
            }

    @server.tool("execute_fallback_query", description="Execute safe fallback queries when main queries fail")
    def mcp_fallback_query(query_type: str = "overview", limit: int = 10):
        """Your exact fallback query patterns from the original GraphBuilder"""
        try:
            config = get_config()
            neo4j_service = Neo4jService(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
            
            # Your exact fallback patterns
            if query_type == "overview":
                query = """
                MATCH (q:Quarter)-[:HAS_METRIC]->(m:Metric)
                RETURN q.period as quarter, m.name as metric, m.value as value, m.unit as unit
                ORDER BY q.period DESC
                LIMIT $limit
                """
            elif query_type == "segments":
                query = """
                MATCH (q:Quarter)-[:HAS_SEGMENT_PERFORMANCE]->(s:Segment)
                RETURN q.period as quarter, s.name as segment, s.revenue as revenue, s.margin as margin
                ORDER BY q.period DESC, s.revenue DESC
                LIMIT $limit
                """
            elif query_type == "ratios":
                query = """
                MATCH (q:Quarter)-[:HAS_RATIO]->(r:Ratio)
                RETURN q.period as quarter, r.name as ratio, r.value as value, r.unit as unit
                ORDER BY q.period DESC
                LIMIT $limit
                """
            else:
                return {
                    "status": "error",
                    "message": f"Unknown fallback query type: {query_type}. Use 'overview', 'segments', or 'ratios'"
                }
            
            results = neo4j_service.execute(query, {"limit": limit})
            neo4j_service.close()
            
            return {
                "status": "success",
                "message": f"Fallback query executed: {query_type}",
                "query_type": query_type,
                "results": results,
                "result_count": len(results)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Fallback query failed: {str(e)}",
                "error_details": str(e)
            }
    
    @server.tool("generate_cypher_query", description="Generate Cypher query from natural language using LLM")
    def mcp_generate_cypher(question: str, limit: int = 10):
        """Generate Cypher query without executing"""
        try:
            config = get_config()
            api_key = config.get_api_key_for_model(config.default_model)
            
            builder = GraphBuilder(
                config.neo4j_uri,
                config.neo4j_user,
                config.neo4j_password, 
                api_key,
                config.default_model
            )
            
            async def generate_async():
                return await builder._generate_query_with_llm(question, limit)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, generate_async())
                cypher_query = future.result(timeout=config.graph_query_timeout)
            
            builder.neo4j.close()
            
            return {
                "status": "success",
                "message": "Cypher query generated successfully",
                "question": question,
                "generated_cypher": cypher_query,
                "model_used": config.default_model,
                "note": "Query generated but not executed"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Cypher generation failed: {str(e)}",
                "error_details": str(e),
                "question": question
            }

    '''@server.tool("execute_raw_cypher", description="Execute raw Cypher queries directly on the graph")
    def mcp_execute_cypher(cypher_query: str, parameters: dict = None):
        """Execute raw Cypher query for advanced users"""
        try:
            config = get_config()
            neo4j_service = Neo4jService(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
            
            # Basic security check - only allow read queries
            cypher_upper = cypher_query.strip().upper()
            if not (cypher_upper.startswith('MATCH') or cypher_upper.startswith('RETURN') or cypher_upper.startswith('WITH')):
                return {
                    "status": "error",
                    "message": "Only read queries (MATCH, RETURN, WITH) are allowed for security",
                    "query_attempted": cypher_query
                }
            
            # Prevent destructive operations
            forbidden_keywords = ['DELETE', 'REMOVE', 'DROP', 'CREATE', 'MERGE', 'SET']
            for keyword in forbidden_keywords:
                if keyword in cypher_upper:
                    return {
                        "status": "error",
                        "message": f"Query contains forbidden keyword: {keyword}. Only read operations allowed.",
                        "query_attempted": cypher_query
                    }
            
            results = neo4j_service.execute(cypher_query, parameters or {})
            neo4j_service.close()
            
            return {
                "status": "success",
                "message": "Raw Cypher query executed successfully",
                "cypher_query": cypher_query,
                "parameters": parameters,
                "results": results,
                "result_count": len(results)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Raw Cypher execution failed: {str(e)}",
                "error_details": str(e),
                "query_attempted": cypher_query
            }
'''