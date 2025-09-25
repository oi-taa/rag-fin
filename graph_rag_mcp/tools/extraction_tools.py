"""
Entity extraction tools for Graph RAG MCP Server
Consolidated extraction functionality preserving all original features
"""

import time
import logging
import asyncio
import concurrent.futures
from typing import Optional
from fastmcp import FastMCP
from config.settings import get_config
from models.financial_models import FinancialChunk, ExtractedEntities
from services.extraction_service import EntityExtractor
from services.neo4j_service import Neo4jService

logger = logging.getLogger(__name__)

def register_extraction_tools(server: FastMCP):
    """Register entity extraction tools"""
    
    # Internal function that does the actual work
    async def _extract_entities_core(text: str, period: Optional[str] = None, model: Optional[str] = None):
        """Core extraction logic that can be called by both tools"""
        try:
            config = get_config()
            model_name = model or config.default_model
            api_key = config.get_api_key_for_model(model_name)
            
            # Validate model availability
            if not api_key and "llama" not in model_name.lower():
                return {
                    "status": "error",
                    "message": f"No API key configured for model: {model_name}",
                    "model_attempted": model_name
                }
            
            # Create chunk for extraction
            chunk_data = {
                "id": f"extract_request_{int(time.time())}",
                "period": period or "Q1_FY2024",
                "type": "extraction_request",
                "size": len(text),
                "text": text
            }
            
            # Validate chunk format
            try:
                chunk = FinancialChunk.model_validate(chunk_data)
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Invalid chunk data: {str(e)}",
                    "chunk_validation_error": str(e)
                }
            
            # Create extractor and extract entities
            extractor = EntityExtractor(model_name, api_key)
            logger.info(f"Extracting entities using model: {extractor.current_model}")
            
            start_time = time.time()
            entities = await extractor.extract(chunk)
            extraction_time = time.time() - start_time
            
            # Convert to dict for JSON response
            entities_dict = entities.dict()
            
            return {
                "status": "success",
                "message": "Financial entities extracted successfully",
                "extraction_info": {
                    "model_used": extractor.current_model,
                    "extraction_time_seconds": round(extraction_time, 2),
                    "text_length": len(text),
                    "input_period": chunk.period
                },
                "entities": entities_dict,
                "entity_counts": {
                    "financial_metrics": len(entities.financial_metrics),
                    "business_segments": len(entities.business_segments),
                    "financial_ratios": len(entities.financial_ratios),
                    "balance_sheet_items": len(entities.balance_sheet_items),
                    "total_entities": len(entities.financial_metrics) + len(entities.business_segments) + len(entities.financial_ratios) + len(entities.balance_sheet_items)
                }
            }
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {
                "status": "error",
                "message": f"Entity extraction failed: {str(e)}",
                "error_details": str(e),
                "model_attempted": model_name if 'model_name' in locals() else "unknown"
            }

    @server.tool("extract_financial_entities", description="Extract financial entities from quarterly report text")
    async def mcp_extract_entities(text: str, period: Optional[str] = None, model: Optional[str] = None):
        """Extract structured financial entities from text using your exact EntityExtractor"""
        return await _extract_entities_core(text=text, period=period, model=model)

    @server.tool("test_entity_extraction", description="Test entity extraction with built-in sample data")
    async def mcp_test_extraction():
        """Test entity extraction using sample ICICI Bank data"""
        
        sample_text = """ICICI Bank Limited Q1_FY2024 NET PROFIT PROFITABILITY ANALYSIS:

NET PROFIT: ₹10,636 crore (+44.0% YoY growth)
Operating Profit: ₹15,660 crore
Net Margin: 20.4% | Operating Margin: 30.1%

INCOME: Total ₹52,084 crore (+32.8% YoY)
Interest Income: ₹37,106 crore (71.2%)
Other Income: ₹14,978 crore (28.8%)

EXPENSES: Total ₹36,424 crore
Interest: ₹16,368 crore | Operating: ₹20,057 crore
Provisions: ₹1,345 crore | Cost Ratio: 69.9%

RETAIL BANKING SEGMENT:
• Revenue: ₹31,057 crore (35.5%)
• Segment Result: ₹4,180 crore
• Margin: 13.5%

TREASURY SEGMENT:
• Revenue: ₹26,306 crore (30.1%)
• Segment Result: ₹4,363 crore
• Margin: 16.6%

BALANCE SHEET:
• Advances: ₹1,124,875 crore (55.1% of total assets)
• Investments: ₹692,709 crore (34.0% of total assets)
• Customer Deposits: ₹1,269,343 crore
• Total Assets: ₹2,039,897 crore
• Total Equity: ₹225,150 crore

RATIOS:
• Basic EPS: ₹15.22 per share (+43.3% YoY)
• Diluted EPS: ₹14.91 per share"""
        
        return await _extract_entities_core(text=sample_text, period="Q1_FY2024", model=None)
        """Test entity extraction using sample ICICI Bank data"""
        
        sample_text = """ICICI Bank Limited Q1_FY2024 NET PROFIT PROFITABILITY ANALYSIS:

NET PROFIT: ₹10,636 crore (+44.0% YoY growth)
Operating Profit: ₹15,660 crore
Net Margin: 20.4% | Operating Margin: 30.1%

INCOME: Total ₹52,084 crore (+32.8% YoY)
Interest Income: ₹37,106 crore (71.2%)
Other Income: ₹14,978 crore (28.8%)

EXPENSES: Total ₹36,424 crore
Interest: ₹16,368 crore | Operating: ₹20,057 crore
Provisions: ₹1,345 crore | Cost Ratio: 69.9%

RETAIL BANKING SEGMENT:
• Revenue: ₹31,057 crore (35.5%)
• Segment Result: ₹4,180 crore
• Margin: 13.5%

TREASURY SEGMENT:
• Revenue: ₹26,306 crore (30.1%)
• Segment Result: ₹4,363 crore
• Margin: 16.6%

BALANCE SHEET:
• Advances: ₹1,124,875 crore (55.1% of total assets)
• Investments: ₹692,709 crore (34.0% of total assets)
• Customer Deposits: ₹1,269,343 crore
• Total Assets: ₹2,039,897 crore
• Total Equity: ₹225,150 crore

RATIOS:
• Basic EPS: ₹15.22 per share (+43.3% YoY)
• Diluted EPS: ₹14.91 per share"""
        
        return await _extract_entities_core(text=sample_text, period="Q1_FY2024", model=None)

    @server.tool("extract_and_save_to_graph", description="Complete pipeline: extract entities from text and save to graph")
    def mcp_extract_and_save(text: str, period: Optional[str] = None, model: Optional[str] = None, dataset_id: str = "mcp_pipeline"):
        """Complete pipeline: extract financial entities and save directly to Neo4j graph"""
        try:
            # Step 1: Extract entities using the core function
            config = get_config()
            model_name = model or config.default_model
            api_key = config.get_api_key_for_model(model_name)
            
            # Validate model availability
            if not api_key and "llama" not in model_name.lower():
                return {
                    "status": "error",
                    "message": f"No API key configured for model: {model_name}",
                    "model_attempted": model_name
                }
            
            # Create chunk for extraction
            chunk_data = {
                "id": f"pipeline_extract_{int(time.time())}",
                "period": period or "Q1_FY2024",
                "type": "pipeline_extraction",
                "size": len(text),
                "text": text
            }
            
            chunk = FinancialChunk.model_validate(chunk_data)
            
            # Extract entities
            extractor = EntityExtractor(model_name, api_key)
            
            async def extract_async():
                return await extractor.extract(chunk)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, extract_async())
                entities = future.result(timeout=config.entity_extraction_timeout)
            
            # Step 2: Save to graph
            if not entities.quarter:
                return {
                    "status": "error",
                    "message": "No quarter found in extracted entities"
                }
            
            neo4j_service = Neo4jService(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
            chunk_id = f"pipeline_{int(time.time())}"
            neo4j_service.save_entities(entities, chunk_id, dataset_id)
            stats = neo4j_service.get_stats()
            neo4j_service.close()
            
            return {
                "status": "success",
                "message": "Complete pipeline successful: extracted entities and saved to graph",
                "pipeline_info": {
                    "model_used": extractor.current_model,
                    "entities_extracted": len(entities.financial_metrics) + len(entities.business_segments) + len(entities.financial_ratios) + len(entities.balance_sheet_items),
                    "quarter": entities.quarter,
                    "dataset_id": dataset_id
                },
                "entity_counts": {
                    "financial_metrics": len(entities.financial_metrics),
                    "business_segments": len(entities.business_segments),
                    "financial_ratios": len(entities.financial_ratios),
                    "balance_sheet_items": len(entities.balance_sheet_items)
                },
                "graph_stats": stats
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Pipeline failed: {str(e)}",
                "error_details": str(e)
            }