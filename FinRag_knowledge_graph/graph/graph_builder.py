# graph/graph_builder.py - Updated with model swapping while preserving all functionality
import logging
import asyncio
import re
from typing import List, Dict, Any
from shared.models import FinancialChunk
from graph.neo4j_service import Neo4jService
from entity.extraction import EntityExtractor

logger = logging.getLogger(__name__)

class GraphBuilder:
    """LLM-powered graph builder with model swapping"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 api_key: str, model_name: str = "gemini-2.0-flash"):
        self.neo4j = Neo4jService(neo4j_uri, neo4j_user, neo4j_password)
        self.extractor = EntityExtractor(model_name, api_key)
        self.current_model = model_name
        
    
    async def build(self, chunks: List[FinancialChunk], dataset_id: str, 
               clear_existing: bool = False) -> Dict[str, Any]:
        """Build knowledge graph from financial chunks - input through api"""
        
        if clear_existing:
            self.neo4j.clear_data(dataset_id)
        
        processed = failed = total_entities = 0
        failed_chunks = []
        
        for chunk in chunks:
            try:
                logger.info(f"Processing chunk {chunk.id} - {chunk.type}")
                entities = await self.extractor.extract(chunk)
                
                # Extraction Log
                logger.info(f"Extracted from {chunk.id}: quarter={entities.quarter}, "
                        f"metrics={len(entities.financial_metrics)}, "
                        f"segments={len(entities.business_segments)}, "
                        f"ratios={len(entities.financial_ratios)}")
                
                if not entities.quarter:
                    logger.warning(f"Chunk {chunk.id} failed: No quarter extracted")
                    failed += 1
                    failed_chunks.append(chunk.id)
                    continue
                
                self.neo4j.save_entities(entities, chunk.id, dataset_id)
                processed += 1
                total_entities += sum([len(entities.financial_metrics), len(entities.business_segments),
                                    len(entities.financial_ratios), len(entities.balance_sheet_items)])
            except Exception as e:
                logger.error(f"Failed chunk {chunk.id}: {e}")
                failed += 1
                failed_chunks.append(chunk.id)
        
        return {"success": True, "chunks_processed": processed, "chunks_failed": failed,
                "total_entities_created": total_entities, "dataset_id": dataset_id,
                "failed_chunk_ids": failed_chunks}
        
    async def query(self, question: str, limit: int = 10) -> List[Dict]:
        """LLM query generation"""
        
        try:
            # Generate Cypher query directly with LLM
            cypher = await self._generate_query_with_llm(question, limit)
            logger.info(f"LLM Generated Query: {cypher}")
            
            # Execute query
            results = self.neo4j.execute(cypher)
            logger.info(f"Query returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            # Fallback query on error
            return self.neo4j.execute(self._fallback_query(limit))
    
    def switch_extraction_model(self, model_name: str, api_key: str = None):
        # Get API key fro config
        if not api_key:
            from shared.config import get_settings
            settings = get_settings()
            api_key = settings.get_api_key_for_model(model_name)
        
        # Switch the model
        self.extractor.switch_model(model_name, api_key)
        self.current_model = model_name
        logger.info(f"Switched extraction model to: {model_name}")
    
    async def _generate_query_with_llm(self, question: str, limit: int) -> str:
        """LLM generates Cypher query directly"""
        
    
        # DEBUG: Check what model extractor is using
        print(f"🔍 Query Generation DEBUG:")
        print(f"   Current model: {self.current_model}")
        print(f"   Extractor client type: {type(self.extractor.client)}")
        if hasattr(self.extractor.client, 'use_groq'):
            print(f"   use_groq: {self.extractor.client.use_groq}")
            print(f"   api_key: {self.extractor.client.api_key}")
    
        
        prompt = f"""Generate a Cypher query for this ICICI Bank financial question: "{question}"

SCHEMA:
Nodes: Organization, Quarter, Metric, Segment, Ratio, BalanceSheetItem

Node Properties:
- Quarter: {{period: "Q1_FY2024", year: 2024, quarter_num: 1}}
- Metric: {{name: "NET PROFIT", value: 10636.0, growth_yoy: 44.0, unit: "crore", quarter: "Q1_FY2024"}}
- Segment: {{name: "RETAIL BANKING SEGMENT", revenue: 31057.0, margin: 13.5, quarter: "Q1_FY2024"}}
- Ratio: {{name: "Cost Ratio", value: 69.9, unit: "percentage", quarter: "Q1_FY2024"}}
- BalanceSheetItem: {{name: "Advances", value: 1124875.0, unit: "crore", quarter: "Q1_FY2024"}}

Relationships:
- HAS_QUARTER: Organization → Quarter
- HAS_METRIC: Quarter → Metric
- HAS_SEGMENT_PERFORMANCE: Quarter → Segment
- HAS_RATIO: Quarter → Ratio
- HAS_BALANCE_SHEET_ITEM: Quarter → BalanceSheetItem

QUARTERS: "Q1_FY2024", "Q2_FY2024", "Q3_FY2024", "Q4_FY2024"

EXACT ENTITY NAMES (use ONLY these):
RATIOS: "Cost Ratio", "Net Margin", "Operating Margin", "Basic EPS", "Diluted EPS"
METRICS: "NET PROFIT", "Operating Profit", "Total Income", "Interest Income", "Other Income", "Total Expenses", "Interest Expenses", "Operating Expenses", "Provisions"
SEGMENTS: "RETAIL BANKING SEGMENT", "WHOLESALE BANKING SEGMENT", "TREASURY SEGMENT", "LIFE INSURANCE SEGMENT", "OTHERS SEGMENT"
BALANCE SHEET: "Advances", "Investments", "Customer Deposits", "Total Assets", "Total Equity"

QUERY PATTERNS - KEEP SIMPLE:

Single Value:
MATCH (q:Quarter {{period: "Q1_FY2024"}})-[:HAS_RATIO]->(r:Ratio {{name: "Cost Ratio"}})
RETURN q.period, r.value, r.unit

Trend Analysis (IMPORTANT - Use This Pattern):
MATCH (q:Quarter)-[:HAS_RATIO]->(r:Ratio {{name: "Cost Ratio"}})
RETURN q.period, r.value
ORDER BY q.period

Growth Calculation:
MATCH (q1:Quarter {{period: "Q1_FY2024"}})-[:HAS_METRIC]->(m1:Metric {{name: "NET PROFIT"}})
MATCH (q4:Quarter {{period: "Q4_FY2024"}})-[:HAS_METRIC]->(m4:Metric {{name: "NET PROFIT"}})
RETURN q1.period, m1.value AS Q1_Value, q4.period, m4.value AS Q4_Value, round(((m4.value - m1.value) / m1.value * 100), 2) AS Growth_Pct

All Segments:
MATCH (q:Quarter {{period: "Q1_FY2024"}})-[:HAS_SEGMENT_PERFORMANCE]->(s:Segment)
RETURN s.name, s.revenue, s.margin
ORDER BY s.revenue DESC

CRITICAL RULES:
1. For "trend", "across quarters", "over time": Use single MATCH with ORDER BY q.period
2. For specific quarter: Use {{period: "Q1_FY2024"}}
3. For comparison: Use two separate MATCH statements for different quarters
4. Use ONLY entity names from the lists above
5. Never match all quarters separately unless doing specific comparisons

QUESTION TYPES:
- "trend across quarters" → MATCH (q:Quarter)-[:HAS_X]->(entity) ORDER BY q.period
- "in Q1 FY2024" → MATCH (q:Quarter {{period: "Q1_FY2024"}})-[:HAS_X]->(entity)
- "compare Q1 and Q4" → Two MATCH statements with q1 and q4
- "all segments" → MATCH with specific quarter, ORDER BY revenue/margin

COLUMN NAMING RULES:
- Use meaningful aliases with AS keyword
- Never return raw property names like "bsi.value" or "m.value"
- Use descriptive names that match the business context

GOOD Examples:
- bsi.value AS Advances_Amount
- m.value AS Net_Profit_Value  
- r.value AS Cost_Ratio_Percentage
- s.revenue AS Segment_Revenue
- q.period AS Quarter

BAD Examples:
- bsi.value (no alias)
- m.value (unclear what metric)
- r.value (unclear what ratio)

ALWAYS use AS with descriptive names that include the entity type and measure.

RETURN ONLY THE CYPHER QUERY - Please don't give any explanations. NO EXPLANATIONS :

Query for: "{question}\""""
        
        try:
            response = await self.extractor.client.generate_content(prompt)
            
            # Clean the response
            cypher = response.strip()
            cypher = re.sub(r'```cypher\n?|```sql\n?|```\n?', '', cypher)
            cypher = cypher.strip()
            
            # Basic validation
            if not cypher.upper().startswith('MATCH') and not cypher.upper().startswith('WITH'):
                logger.warning(f"Generated query doesn't start with MATCH: {cypher}")
                return self._fallback_query(limit)
            
            return cypher
            
        except Exception as e:
            logger.error(f"LLM query generation failed: {e}")
            return self._fallback_query(limit)

    def _fallback_query(self, limit: int) -> str:
        """Safe fallback query - UNCHANGED"""
        return f"""
            MATCH (q:Quarter)-[:HAS_METRIC]->(m:Metric)
            RETURN q.period as quarter, m.name as metric, m.value as value, m.unit as unit
            ORDER BY q.period DESC
            LIMIT {limit}
        """
   
    # Utility methods - UNCHANGED
    def is_healthy(self) -> bool: 
        return self.neo4j.health_check()
    
    def get_stats(self) -> Dict: 
        return self.neo4j.get_stats()
    
    def clear(self, dataset_id: str): 
        self.neo4j.clear_data(dataset_id)
    
    
    def get_current_extraction_model(self) -> str:
        """Get current extraction model info"""
        return self.current_model