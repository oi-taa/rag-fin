"""
Graph management tools for Graph RAG MCP Server
Preserving all original graph building and management functionality
"""

import time
import logging
import asyncio
import concurrent.futures
from typing import List, Dict
from fastmcp import FastMCP
from config.settings import get_config
from models.financial_models import FinancialChunk
from services.neo4j_service import Neo4jService
from services.extraction_service import EntityExtractor


logger = logging.getLogger(__name__)

def register_graph_tools(server: FastMCP):
    """Register graph management tools"""
    
    @server.tool("build_financial_graph", description="Build knowledge graph from financial chunks")
    def mcp_build_graph(chunks: List[Dict], dataset_id: str = "financial_data", clear_existing: bool = False):
        """Build graph from chunks - auto-detects text vs structured format"""
        try:
            config = get_config()
            neo4j_service = Neo4jService(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
            
            if clear_existing:
                neo4j_service.clear_data(dataset_id)
            
            processed = failed = total_entities = 0
            failed_chunks = []
            
            for chunk_data in chunks:
                try:
                    # Smart format detection and processing
                    result = safe_chunk_processing(chunk_data, config)
                    
                    if result["status"] == "success":
                        entities = result["entities"]
                        chunk_id = result["chunk_id"]
                        company_name = result.get("company_name", "ICICI Bank")
                        
                        # Validate entities
                        if not entities.quarter:
                            failed += 1
                            failed_chunks.append(chunk_id)
                            continue
                        
                        # Save to graph
                        neo4j_service.save_entities(entities, chunk_id, dataset_id, company_name)
                        processed += 1
                        total_entities += len(entities.financial_metrics) + len(entities.business_segments) + len(entities.financial_ratios) + len(entities.balance_sheet_items)
                        
                    else:
                        failed += 1
                        failed_chunks.append(result.get("chunk_id", "unknown"))
                        logger.error(f"Chunk processing failed: {result['message']}")
                        
                except Exception as e:
                    logger.error(f"Failed chunk {chunk_data.get('id', chunk_data.get('company', 'unknown'))}: {e}")
                    failed += 1
                    failed_chunks.append(chunk_data.get('id', chunk_data.get('company', 'unknown')))
            
            stats = neo4j_service.get_stats()
            neo4j_service.close()
            
            return {
                "status": "success",
                "message": "Graph built successfully with smart format detection",
                "build_result": {
                    "chunks_processed": processed,
                    "chunks_failed": failed,
                    "total_entities_created": total_entities,
                    "dataset_id": dataset_id,
                    "failed_chunk_ids": failed_chunks
                },
                "graph_stats": stats
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Graph building failed: {str(e)}",
                "error_details": str(e)
            }

    def safe_chunk_processing(chunk_data: Dict, config) -> Dict:
        """Process chunk regardless of format - auto-detection"""
        
        # Try text format first
        try:
            if ("text" in chunk_data and 
                "id" in chunk_data and 
                "period" in chunk_data and 
                "type" in chunk_data and 
                "size" in chunk_data):
                
                logger.info(f"Detected text format chunk: {chunk_data['id']}")
                
                # Validate with Pydantic
                chunk = FinancialChunk.model_validate(chunk_data)
                
                # Extract entities using LLM
                extractor = EntityExtractor(config.default_model, config.get_api_key_for_model(config.default_model))
                
                import concurrent.futures
                async def extract_async():
                    return await extractor.extract(chunk)
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, extract_async())
                    entities = future.result(timeout=config.entity_extraction_timeout)
                
                return {
                    "status": "success",
                    "entities": entities,
                    "chunk_id": chunk.id,
                    "format_detected": "text"
                }
                
        except Exception as e:
            logger.debug(f"Text format processing failed: {e}")
        
        # Try structured format
        try:
            if "financialResults" in chunk_data:
                logger.info(f"Detected structured format chunk: {chunk_data.get('company', 'unknown')}")
                
                # Import the converter function
                from services.extraction_service import convert_structured_to_entities
                
                # Convert directly to entities (no LLM needed) - now returns tuple
                entities, company_name = convert_structured_to_entities(chunk_data)
                chunk_id = chunk_data.get("company", f"structured_{int(time.time())}")
                
                return {
                    "status": "success",
                    "entities": entities,
                    "chunk_id": chunk_id,
                    "company_name": company_name,  # Add company name to result
                    "format_detected": "structured"
                }
            
        except Exception as e:
            logger.error(f"Structured format processing failed: {e}")
        
        # Neither format worked
        return {
            "status": "error",
            "message": f"Unknown chunk format. Expected text format (id, period, type, size, text) or structured format (financialResults). Got: {list(chunk_data.keys())}",
            "chunk_id": chunk_data.get('id', chunk_data.get('company', 'unknown_format')),
            "format_detected": "unknown"
        }

    @server.tool("get_graph_stats", description="Get Neo4j knowledge graph statistics")
    def mcp_get_graph_stats():
        """Get comprehensive Neo4j graph statistics"""
        try:
            config = get_config()
            neo4j_service = Neo4jService(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
            
            stats = neo4j_service.get_stats()
            
            neo4j_service.close()
            
            return {
                "status": "success", 
                "message": "Graph statistics retrieved successfully",
                "graph_stats": stats
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to get graph stats: {str(e)}",
                "error_details": str(e)
            }

    @server.tool("clear_graph_data", description="Clear data from Neo4j knowledge graph")
    def mcp_clear_graph(dataset_id: str = None):
        """Clear graph data by dataset ID or all ICICI data"""
        try:
            config = get_config()
            neo4j_service = Neo4jService(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
            
            # Clear data
            neo4j_service.clear_data(dataset_id)
            
            # Get updated stats
            stats = neo4j_service.get_stats()
            
            neo4j_service.close()
            
            return {
                "status": "success",
                "message": f"Graph data cleared successfully",
                "clear_info": {
                    "dataset_id": dataset_id or "all_icici_data",
                    "scope": "specific_dataset" if dataset_id else "all_data"
                },
                "updated_graph_stats": stats
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Failed to clear graph data: {str(e)}",
                "error_details": str(e)
            }

    '''@server.tool("traverse_entity_relationships", description="Find related entities across the knowledge graph")
    def mcp_traverse_relationships(entity_name: str, entity_type: str, max_depth: int = 2):
        """Find all entities connected to a specific entity"""
        try:
            config = get_config()
            neo4j_service = Neo4jService(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
            
            # Normalize entity type
            entity_type_clean = entity_type.lower().strip()
            
            # Build traversal query based on entity type
            if entity_type_clean == "metric":
                query = """
                MATCH (m:Metric {name: $entity_name})
                MATCH (m)<-[:HAS_METRIC]-(q:Quarter)
                OPTIONAL MATCH (q)-[:HAS_SEGMENT_PERFORMANCE]->(s:Segment)
                OPTIONAL MATCH (q)-[:HAS_RATIO]->(r:Ratio)
                RETURN q.period as quarter, 
                       collect(DISTINCT s.name) as related_segments,
                       collect(DISTINCT r.name) as related_ratios
                ORDER BY q.period
                """
            elif entity_type_clean == "segment":
                query = """
                MATCH (s:Segment {name: $entity_name})
                MATCH (s)<-[:HAS_SEGMENT_PERFORMANCE]-(q:Quarter)
                OPTIONAL MATCH (q)-[:HAS_METRIC]->(m:Metric)
                OPTIONAL MATCH (q)-[:HAS_RATIO]->(r:Ratio)
                RETURN q.period as quarter,
                       collect(DISTINCT m.name) as related_metrics,
                       collect(DISTINCT r.name) as related_ratios
                ORDER BY q.period
                """
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported entity type: '{entity_type}'. Use 'metric' or 'segment'",
                    "received_type": entity_type,
                    "cleaned_type": entity_type_clean
                }
            
            results = neo4j_service.execute(query, {"entity_name": entity_name})
            neo4j_service.close()
            
            return {
                "status": "success",
                "message": f"Found relationships for {entity_name}",
                "entity": {"name": entity_name, "type": entity_type},
                "relationships": results,
                "relationship_count": len(results)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Relationship traversal failed: {str(e)}",
                "error_details": str(e)
            }

    @server.tool("compare_quarters", description="Complex comparison analysis across multiple quarters")
    def mcp_compare_quarters(quarters: List[str], comparison_type: str = "growth"):
        """Advanced multi-quarter comparison with growth analysis"""
        try:
            config = get_config()
            neo4j_service = Neo4jService(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
            
            if comparison_type == "growth":
                query = """
                MATCH (q:Quarter)-[:HAS_METRIC]->(m:Metric)
                WHERE q.period IN $quarters
                WITH m.name as metric_name, 
                     collect({quarter: q.period, value: m.value}) as quarter_values
                WHERE size(quarter_values) > 1
                RETURN metric_name,
                       quarter_values,
                       reduce(total = 0, qv in quarter_values | total + qv.value) as total_value
                ORDER BY total_value DESC
                """
            elif comparison_type == "segments":
                query = """
                MATCH (q:Quarter)-[:HAS_SEGMENT_PERFORMANCE]->(s:Segment)
                WHERE q.period IN $quarters
                WITH s.name as segment_name,
                     collect({quarter: q.period, revenue: s.revenue, margin: s.margin}) as performance
                RETURN segment_name, performance
                ORDER BY segment_name
                """
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported comparison type: {comparison_type}. Use 'growth' or 'segments'"
                }
            
            results = neo4j_service.execute(query, {"quarters": quarters})
            neo4j_service.close()
            
            # Calculate growth rates for growth comparison
            if comparison_type == "growth" and results:
                for result in results:
                    quarter_values = result.get("quarter_values", [])
                    if len(quarter_values) >= 2:
                        # Sort by quarter to calculate sequential growth
                        sorted_quarters = sorted(quarter_values, key=lambda x: x["quarter"])
                        growth_rates = []
                        for i in range(1, len(sorted_quarters)):
                            prev_val = sorted_quarters[i-1]["value"]
                            curr_val = sorted_quarters[i]["value"]
                            if prev_val and prev_val != 0:
                                growth_rate = ((curr_val - prev_val) / prev_val) * 100
                                growth_rates.append({
                                    "from": sorted_quarters[i-1]["quarter"],
                                    "to": sorted_quarters[i]["quarter"],
                                    "growth_rate": round(growth_rate, 2)
                                })
                        result["growth_analysis"] = growth_rates
            
            return {
                "status": "success",
                "message": f"Quarter comparison completed for {len(quarters)} quarters",
                "comparison_type": comparison_type,
                "quarters_analyzed": quarters,
                "results": results
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Quarter comparison failed: {str(e)}",
                "error_details": str(e)
            }
'''