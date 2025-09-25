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
import json
import logging
import asyncio
import time
import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import lru_cache
from neo4j import GraphDatabase
# External dependencies
import aiohttp
from dotenv import load_dotenv
from fastmcp import FastMCP

# We need pydantic for exact compatibility with your models
try:
    from pydantic import BaseModel, Field
    from enum import Enum
except ImportError:
    logger.error("pydantic not installed. Install with: pip install pydantic")
    sys.exit(1)

# Load environment variables
load_dotenv()

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -------------------------
# Configuration
# -------------------------
@dataclass
class GraphRAGConfig:
    """Configuration for Graph RAG MCP Server"""
    # Neo4j Configuration
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
    
    # LLM API Keys
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    
    
    # Default Settings
    default_model: str = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash")
    
    base_graph_build_timeout: int = int(os.getenv("GRAPH_BUILD_TIMEOUT", "300"))
    base_entity_extraction_timeout: int = int(os.getenv("ENTITY_EXTRACTION_TIMEOUT", "60"))
    base_graph_query_timeout: int = int(os.getenv("GRAPH_QUERY_TIMEOUT", "30"))
    
    
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Server Configuration
    port: int = int(os.getenv("GRAPH_RAG_PORT", "9008"))
    host: str = os.getenv("GRAPH_RAG_HOST", "0.0.0.0")
    @property
    def graph_build_timeout(self) -> int:
        """Get timeout based on current model"""
        if "llama" in self.default_model.lower():
            return self.base_graph_build_timeout * 2
        return self.base_graph_build_timeout
    
    @property 
    def entity_extraction_timeout(self) -> int:
        """Get extraction timeout based on current model"""
        if "llama" in self.default_model.lower():
            return self.base_entity_extraction_timeout * 2
        return self.base_entity_extraction_timeout
    
    @property
    def graph_query_timeout(self) -> int:
        """Get query timeout based on current model"""
        if "llama" in self.default_model.lower():
            return self.base_graph_query_timeout * 2
        return self.base_graph_query_timeout
    
    def get_api_key_for_model(self, model_name: str) -> str:
        """Get appropriate API key for model"""
        if "gemini" in model_name.lower():
            return self.gemini_api_key
        elif "gpt" in model_name.lower() or "openai" in model_name.lower():
            return self.openai_api_key
        elif "groq" in model_name.lower():
            return self.groq_api_key
        elif "llama" in model_name.lower():
            return ""  # Local model, no API key needed
        else:
            return ""

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check Neo4j configuration
        if not self.neo4j_uri:
            issues.append("NEO4J_URI not configured")
        if self.neo4j_password == "password":
            issues.append("Using default Neo4j password - consider changing for security")
        
        # Check API keys
        if not self.gemini_api_key and "gemini" in self.default_model:
            issues.append(f"GEMINI_API_KEY required for default model: {self.default_model}")
        if not self.openai_api_key and "gpt" in self.default_model:
            issues.append(f"OPENAI_API_KEY required for default model: {self.default_model}")
        
        return issues

# Global configuration instance
@lru_cache()
def get_config() -> GraphRAGConfig:
    return GraphRAGConfig()

# -------------------------
# Data Models (EXACT copy from shared/models.py - NO CHANGES)
# -------------------------

# We need pydantic for exact compatibility
try:
    from pydantic import BaseModel, Field
    from enum import Enum
except ImportError:
    logger.error("pydantic not installed. Install with: pip install pydantic")
    sys.exit(1)

class ChunkType(str, Enum):
    BALANCE_SHEET = "balance_sheet_analysis"
    FINANCIAL_RATIOS = "financial_ratios" 
    PROFITABILITY = "profitability_analysis"
    SEGMENT_ANALYSIS = "segment_analysis"

class FinancialChunk(BaseModel):
    """Financial chunk from ICICI quarterly reports - EXACT COPY"""
    id: str
    period: str = Field(..., pattern=r"Q[1-4]_FY\d{4}")  # e.g., Q1_FY2024
    type: str  # Using str instead of enum for flexibility
    size: int
    text: str = Field(..., min_length=10)

class FinancialMetric(BaseModel):
    """Financial metric (profit, revenue, etc.) - EXACT COPY"""
    name: str
    value: float
    growth_yoy: Optional[float] = None
    unit:  Optional[str] = "crore"

class BusinessSegment(BaseModel):
    """Business segment performance - EXACT COPY"""
    name: str
    revenue: float
    margin: float
    percentage_of_total: Optional[float] = None

class FinancialRatio(BaseModel):
    """Financial ratio (EPS, margins, etc.) - EXACT COPY"""
    name: str
    value: float
    growth_yoy: Optional[float] = None
    unit: Optional[str] = "ratio" 

class BalanceSheetItem(BaseModel):
    """Balance sheet item - EXACT COPY"""
    name: str
    value: float
    percentage_of_total: Optional[float] = None
    unit:  Optional[str] = "crore"

class ExtractedEntities(BaseModel):
    """All entities extracted from a chunk - EXACT COPY"""
    quarter: Optional[str] = None
    financial_metrics: List[FinancialMetric] = []
    business_segments: List[BusinessSegment] = []
    financial_ratios: List[FinancialRatio] = []
    balance_sheet_items: List[BalanceSheetItem] = []

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
            cypher = await self._generate_query_with_llm(question, limit)
            logger.info(f"LLM Generated Query: {cypher}")
            results = self.neo4j.execute(cypher)
            logger.info(f"Query returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return self.neo4j.execute(self._fallback_query(limit))
    
    async def _generate_query_with_llm(self, question: str, limit: int) -> str:
        """LLM generates Cypher query directly"""
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
            cypher = response.strip()
            cypher = re.sub(r'```cypher\n?|```sql\n?|```\n?', '', cypher)
            cypher = cypher.strip()
            
            if not cypher.upper().startswith('MATCH') and not cypher.upper().startswith('WITH'):
                logger.warning(f"Generated query doesn't start with MATCH: {cypher}")
                return self._fallback_query(limit)
            
            return cypher
        except Exception as e:
            logger.error(f"LLM query generation failed: {e}")
            return self._fallback_query(limit)

    def _fallback_query(self, limit: int) -> str:
        """Safe fallback query"""
        return f"""
            MATCH (q:Quarter)-[:HAS_METRIC]->(m:Metric)
            RETURN q.period as quarter, m.name as metric, m.value as value, m.unit as unit
            ORDER BY q.period DESC
            LIMIT {limit}
        """
    
class Neo4jService:
    """Fixed Neo4j operations - stores properties on NODES, not relationships"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_constraints()
    
    def _create_constraints(self):
        """Create constraints for all entity types"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (org:Organization) REQUIRE org.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (q:Quarter) REQUIRE q.period IS UNIQUE",  # Changed to period
            "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Metric) REQUIRE (m.name, m.quarter) IS UNIQUE",  # Composite key
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Segment) REQUIRE (s.name, s.quarter) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Ratio) REQUIRE (r.name, r.quarter) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:BalanceSheetItem) REQUIRE (b.name, b.quarter) IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except:
                    pass  # Constraint already exists
    
    def health_check(self) -> bool:
        """Check if Neo4j is accessible"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except:
            return False
    
    def save_entities(self, entities: ExtractedEntities, chunk_id: str, dataset_id: str):
        """Save all entity types with properties stored ON NODES"""
        
        with self.driver.session() as session:
            quarter = entities.quarter
            if not quarter:
                return
            
            # Create quarter and organization - store properties on QUARTER NODE
            session.run("""
                MERGE (org:Organization {name: "ICICI Bank"})
                MERGE (q:Quarter {period: $quarter})
                SET q.dataset_id = $dataset_id, 
                    q.source_chunks = COALESCE(q.source_chunks, []) + [$chunk_id],
                    q.year = CASE 
                        WHEN $quarter CONTAINS "FY2024" THEN 2024
                        WHEN $quarter CONTAINS "FY2025" THEN 2025
                        ELSE 2024 END,
                    q.quarter_num = CASE 
                        WHEN $quarter STARTS WITH "Q1" THEN 1
                        WHEN $quarter STARTS WITH "Q2" THEN 2
                        WHEN $quarter STARTS WITH "Q3" THEN 3
                        WHEN $quarter STARTS WITH "Q4" THEN 4
                        ELSE 1 END
                MERGE (org)-[:HAS_QUARTER]->(q)
            """, quarter=quarter, dataset_id=dataset_id, chunk_id=chunk_id)
            
            # Save all entity types with properties on NODES
            self._save_metrics(session, quarter, entities.financial_metrics)
            self._save_segments(session, quarter, entities.business_segments)
            self._save_ratios(session, quarter, entities.financial_ratios)
            self._save_balance_sheet_items(session, quarter, entities.balance_sheet_items)
    
    def _save_metrics(self, session, quarter: str, metrics: List):
        """Save metrics with properties on NODES"""
        for metric in metrics:
            try:
                session.run("""
                    MATCH (q:Quarter {period: $quarter})
                    MERGE (m:Metric {name: $name, quarter: $quarter})
                    SET m.value = $value,
                        m.growth_yoy = $growth_yoy,
                        m.unit = $unit,
                        m.source_chunk = $source_chunk,
                        m.last_updated = datetime()
                    MERGE (q)-[:HAS_METRIC]->(m)
                """, 
                quarter=quarter, 
                name=metric.name,
                value=metric.value,
                growth_yoy=metric.growth_yoy,
                unit=metric.unit,
                source_chunk=quarter + "_metrics"
                )
            except Exception as e:
                logger.error(f"Failed to save metric {metric.name}: {e}")
    
    def _save_segments(self, session, quarter: str, segments: List):
        """Save segments with properties on NODES"""
        for segment in segments:
            try:
                session.run("""
                    MATCH (q:Quarter {period: $quarter})
                    MERGE (s:Segment {name: $name, quarter: $quarter})
                    SET s.revenue = $revenue,
                        s.margin = $margin,
                        s.percentage_of_total = $percentage_of_total,
                        s.source_chunk = $source_chunk,
                        s.last_updated = datetime()
                    MERGE (q)-[:HAS_SEGMENT_PERFORMANCE]->(s)
                """,
                quarter=quarter,
                name=segment.name,
                revenue=segment.revenue,
                margin=segment.margin,
                percentage_of_total=getattr(segment, 'percentage_of_total', None),
                source_chunk=quarter + "_segments"
                )
            except Exception as e:
                logger.error(f"Failed to save segment {segment.name}: {e}")
    
    def _save_ratios(self, session, quarter: str, ratios: List):
        """Save ratios with properties on NODES"""
        for ratio in ratios:
            try:
                session.run("""
                    MATCH (q:Quarter {period: $quarter})
                    MERGE (r:Ratio {name: $name, quarter: $quarter})
                    SET r.value = $value,
                        r.growth_yoy = $growth_yoy,
                        r.unit = $unit,
                        r.source_chunk = $source_chunk,
                        r.last_updated = datetime()
                    MERGE (q)-[:HAS_RATIO]->(r)
                """,
                quarter=quarter,
                name=ratio.name,
                value=ratio.value,
                growth_yoy=getattr(ratio, 'growth_yoy', None),
                unit=ratio.unit,
                source_chunk=quarter + "_ratios"
                )
            except Exception as e:
                logger.error(f"Failed to save ratio {ratio.name}: {e}")
    
    def _save_balance_sheet_items(self, session, quarter: str, items: List):
        """Save balance sheet items with properties on NODES"""
        for item in items:
            try:
                session.run("""
                    MATCH (q:Quarter {period: $quarter})
                    MERGE (b:BalanceSheetItem {name: $name, quarter: $quarter})
                    SET b.value = $value,
                        b.percentage_of_total = $percentage_of_total,
                        b.unit = $unit,
                        b.source_chunk = $source_chunk,
                        b.last_updated = datetime()
                    MERGE (q)-[:HAS_BALANCE_SHEET_ITEM]->(b)
                """,
                quarter=quarter,
                name=item.name,
                value=item.value,
                percentage_of_total=getattr(item, 'percentage_of_total', None),
                unit=item.unit,
                source_chunk=quarter + "_balance_sheet"
                )
            except Exception as e:
                logger.error(f"Failed to save balance sheet item {item.name}: {e}")
    
    def execute(self, cypher_query: str, parameters: Dict = None) -> List[Dict[str, Any]]:
        """Execute Cypher query and return results"""
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    
    
    def get_stats(self) -> Dict[str, int]:
        """Get comprehensive graph statistics"""
        with self.driver.session() as session:
            try:
                # Count nodes by type
                result = session.run("""
                    MATCH (n) 
                    WITH labels(n)[0] as type, count(n) as count
                    RETURN type, count
                    ORDER BY type
                """)
                stats = {f"{record['type']}_count": record['count'] 
                        for record in result if record['type']}
                
                # Get quarters
                result = session.run("MATCH (q:Quarter) RETURN q.period as quarter ORDER BY quarter")
                stats['quarters_available'] = [record['quarter'] for record in result]
                
                # Get sample data counts
                result = session.run("""
                    MATCH (q:Quarter)
                    OPTIONAL MATCH (q)-[:HAS_METRIC]->(m:Metric)
                    OPTIONAL MATCH (q)-[:HAS_SEGMENT_PERFORMANCE]->(s:Segment)
                    OPTIONAL MATCH (q)-[:HAS_RATIO]->(r:Ratio)
                    OPTIONAL MATCH (q)-[:HAS_BALANCE_SHEET_ITEM]->(b:BalanceSheetItem)
                    RETURN q.period as quarter,
                           count(DISTINCT m) as metrics_count,
                           count(DISTINCT s) as segments_count,
                           count(DISTINCT r) as ratios_count,
                           count(DISTINCT b) as balance_items_count
                    ORDER BY q.period
                """)
                
                stats['detailed_counts'] = {
                    record['quarter']: {
                        'metrics': record['metrics_count'],
                        'segments': record['segments_count'],
                        'ratios': record['ratios_count'],
                        'balance_items': record['balance_items_count']
                    } for record in result
                }
                
                return stats
            except Exception as e:
                logger.error(f"Stats failed: {e}")
                return {"error": str(e)}
    
    def clear_data(self, dataset_id: str = None):
        """Clear data by dataset or all ICICI data"""
        with self.driver.session() as session:
            try:
                if dataset_id:
                    session.run("""
                        MATCH (q:Quarter {dataset_id: $dataset_id})
                        DETACH DELETE q
                    """, dataset_id=dataset_id)
                    session.run("""
                        MATCH (n) WHERE n.source_chunk CONTAINS $dataset_id
                        DETACH DELETE n
                    """, dataset_id=dataset_id)
                else:
                    session.run("MATCH (org:Organization {name: 'ICICI Bank'}) DETACH DELETE org")
                    session.run("MATCH (n) WHERE n.quarter IS NOT NULL DETACH DELETE n")
            except Exception as e:
                logger.error(f"Clear failed: {e}")
    
    def session(self):
        """Direct session access"""
        return self.driver.session()
    
    def close(self):
        """Close connection"""
        if self.driver:
            self.driver.close()

# -------------------------
# Rate Limiter (from your entity/extraction.py)
# -------------------------

class RateLimiter:
    def __init__(self, delay: float = 4.0):
        self.delay = delay
        self.last_call = 0
    
    async def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self.last_call = time.time()

# -------------------------
# Entity Extractor (EXACT copy from entity/extraction.py - NO CHANGES)
# -------------------------

class EntityExtractor:
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = None, **kwargs):
        self.current_model, self.api_key = model_name, api_key
        if "gemini" in model_name: self.client = GeminiProvider(model_name, api_key)
        elif "llama" in model_name: self.client = LlamaProvider(model_name, api_key, **kwargs)
        elif "gpt" in model_name: self.client = GPTProvider(model_name, api_key)
        else: self.client = GeminiProvider("gemini-2.0-flash", api_key)
    
    async def extract(self, chunk: FinancialChunk) -> ExtractedEntities:
        try:
            prompt = self._build_prompt(chunk.text)
            response = await self.client.generate_content(prompt)
            
            if not response or not response.strip():
                return ExtractedEntities()
            
            # Clean response text
            response_text = response.strip()
            response_text = re.sub(r'```json\n?|```\n?', '', response_text)
            
            # Extract JSON from response (handles Llama's explanatory text)
            json_text = self._extract_json_from_response(response_text)
            if not json_text:
                return ExtractedEntities()
            
            # Clean floating point precision issues
            json_text = re.sub(r'(\d+)\.0{20,}', r'\1.0', json_text)
            json_text = re.sub(r'(\d+\.\d{1,2})\d{20,}', r'\1', json_text)
            
            try:
                parsed = json.loads(json_text)
                cleaned_data = self._clean_parsed_data(parsed)
                return ExtractedEntities(**cleaned_data)  # Fixed: use cleaned_data
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                return ExtractedEntities()
                
        except Exception as e:
            logger.error(f"Extraction failed for {chunk.id}: {e}")
            return ExtractedEntities()

    def _extract_json_from_response(self, response_text: str) -> str:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        
        if json_start == -1 or json_end == -1 or json_start >= json_end:
            return ""
        
        return response_text[json_start:json_end + 1]

    def _clean_parsed_data(self, data: dict) -> dict:
        def filter_valid_items(items_list, required_field):
            if not items_list:
                return []
            return [item for item in items_list 
                    if item.get(required_field) is not None]
        
        return {
            'quarter': data.get('quarter'),
            'financial_metrics': filter_valid_items(data.get('financial_metrics', []), 'value'),
            'business_segments': filter_valid_items(data.get('business_segments', []), 'revenue'), 
            'financial_ratios': filter_valid_items(data.get('financial_ratios', []), 'value'),
            'balance_sheet_items': filter_valid_items(data.get('balance_sheet_items', []), 'value')
        }
    
    def switch_model(self, model_name: str, api_key: str = None, **kwargs):
        self.current_model, self.api_key = model_name, api_key or self.api_key
        if "gemini" in model_name: self.client = GeminiProvider(model_name, self.api_key)
        elif "llama" in model_name: self.client = LlamaProvider(model_name, api_key, **kwargs)
        elif "gpt" in model_name: self.client = GPTProvider(model_name, self.api_key)
    
    def _build_prompt(self, text: str) -> str:
        """Build comprehensive extraction prompt for ICICI data"""
        return f"""
        Extract ALL financial data from this ICICI Bank quarterly report text:
        
        {text}
        
        Return JSON with this EXACT schema:
        {{
            "quarter": "Q1_FY2024",
            "financial_metrics": [
                {{"name": "NET PROFIT", "value": 10636.0, "growth_yoy": 44.0, "unit": "crore"}},
                {{"name": "Operating Profit", "value": 15660.0, "unit": "crore"}},
                {{"name": "Total Income", "value": 52084.0, "growth_yoy": 32.8, "unit": "crore"}},
                {{"name": "Interest Income", "value": 37106.0, "unit": "crore"}},
                {{"name": "Other Income", "value": 14978.0, "unit": "crore"}},
                {{"name": "Total Expenses", "value": 36424.0, "unit": "crore"}},
                {{"name": "Interest Expenses", "value": 16368.0, "unit": "crore"}},
                {{"name": "Operating Expenses", "value": 20057.0, "unit": "crore"}},
                {{"name": "Provisions", "value": 1345.0, "unit": "crore"}}
            ],
            "business_segments": [
                {{"name": "RETAIL BANKING SEGMENT", "revenue": 31057.0, "margin": 13.5, "percentage_of_total": 35.5}},
                {{"name": "TREASURY SEGMENT", "revenue": 26306.0, "margin": 16.6, "percentage_of_total": 30.1}}
            ],
            "financial_ratios": [
                {{"name": "Basic EPS", "value": 15.22, "growth_yoy": 43.3, "unit": "per share"}},
                {{"name": "Diluted EPS", "value": 14.91, "unit": "per share"}},
                {{"name": "Net Margin", "value": 20.4, "unit": "percentage"}},
                {{"name": "Operating Margin", "value": 30.1, "unit": "percentage"}},
                {{"name": "Cost Ratio", "value": 69.9, "unit": "percentage"}}
            ],
            "balance_sheet_items": [
                {{"name": "Advances", "value": 1124875.0, "percentage_of_total": 55.1, "unit": "crore"}},
                {{"name": "Investments", "value": 692709.0, "percentage_of_total": 34.0, "unit": "crore"}},
                {{"name": "Customer Deposits", "value": 1269343.0, "unit": "crore"}},
                {{"name": "Total Assets", "value": 2039897.0, "unit": "crore"}},
                {{"name": "Total Equity", "value": 225150.0, "unit": "crore"}}
            ]
        }}

        EXTRACTION RULES:
        1. Extract EVERY financial number mentioned in the text
        2. Convert currency: ₹52,084 crore → 52084.0 
        3. Convert percentages: 20.4% → 20.4
        4. Extract growth rates: (+44.0% YoY) → 44.0
        5. Use exact names from text but standardize format
        6. Include ALL income items, expense items, ratios, segments, balance sheet items
        7. If percentage of total is mentioned, include it
        8. Return null for missing values, don't make up data
        
        INCOME STATEMENT ITEMS TO EXTRACT:
        - Total Income, Interest Income, Other Income
        - Total Expenses, Interest Expenses, Operating Expenses
        - Net Profit, Operating Profit, Provisions
        
        RATIOS TO EXTRACT:
        - EPS (Basic, Diluted), Margins (Net, Operating)
        - Cost Ratio, any other percentages mentioned
        
        BALANCE SHEET TO EXTRACT:
        - Advances, Investments, Deposits, Assets, Equity
        - Cash balances, Borrowings, Reserves
        
        SEGMENT DATA TO EXTRACT:
        - Revenue, Margin, Percentage of total for each segment
        
        Be comprehensive and extract EVERYTHING mentioned!
        Return only valid JSON, NO EXPLANATIONS ONLY VALID JSON!
        """

# Helper function for quick extraction (from your code)
async def quick_extract(chunk_text: str, model: str = "gemini-2.0-flash", api_key: str = None) -> ExtractedEntities:
    """Quick extraction for testing"""
    chunk = FinancialChunk(id="test_chunk", period="Q1_FY2024", type="test", size=len(chunk_text), text=chunk_text)
    extractor = EntityExtractor(model, api_key)
    return await extractor.extract(chunk)

class LLMProvider(ABC):
    def __init__(self, model_name: str, api_key: str = None, rate_limit: float = 1.0):
        self.model_name = model_name
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.last_call = 0
    
    async def _rate_limit_wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_call = time.time()
    
    @abstractmethod
    async def generate_content(self, prompt: str) -> str:
        pass

class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = None, rate_limit: float = 4.0):
        super().__init__(model_name, api_key, rate_limit)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model_name)
        except ImportError:
            logger.error("google-generativeai not installed. Install with: pip install google-generativeai")
            raise
    
    async def generate_content(self, prompt: str) -> str:
        await self._rate_limit_wait()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self.client.generate_content(prompt))
        return response.text

class LlamaProvider(LLMProvider):
    def __init__(self, model_name: str = "llama3.1:8b", api_key: str = None, 
                 base_url: str = "http://localhost:11434", rate_limit: float = 0.5):
        super().__init__(model_name, api_key, rate_limit)
        self.base_url = base_url
        
        self.use_groq = api_key is not None and api_key.strip() != ""
        
        logger.debug(f"🔍 LlamaProvider initialized:")
        logger.debug(f"   api_key: '{api_key}'")
        logger.debug(f"   api_key is not None: {api_key is not None}")
        logger.debug(f"   api_key.strip() != '': {api_key.strip() != '' if api_key else False}")
        logger.debug(f"   use_groq: {self.use_groq}")
    
    async def generate_content(self, prompt: str) -> str:
        await self._rate_limit_wait()
        
        if self.use_groq:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": "llama-3.1-70b-versatile",
                "temperature": 0.1,
                "max_tokens": 8192
            }
            async with aiohttp.ClientSession() as session:
                async with session.post("https://api.groq.com/openai/v1/chat/completions", 
                                      headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        raise Exception(f"Groq error: {response.status}")
        else:
            payload = {"model": self.model_name, "prompt": prompt, "stream": False}
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        raise Exception(f"Ollama error: {response.status}")

class GPTProvider(LLMProvider):
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None, rate_limit: float = 1.0):
        super().__init__(model_name, api_key, rate_limit)
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            logger.error("openai not installed. Install with: pip install openai")
            raise
    
    async def generate_content(self, prompt: str) -> str:
        await self._rate_limit_wait()
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=8192
        )
        return response.choices[0].message.content

class ModelFactory:
    @staticmethod
    def create_provider(model_type: str, api_key: str = None, **kwargs) -> LLMProvider:
        providers = {"gemini": GeminiProvider, "llama": LlamaProvider, "gpt": GPTProvider}
        if model_type not in providers:
            raise ValueError(f"Unknown model: {model_type}")
        return providers[model_type](api_key=api_key, **kwargs)
FINANCIAL_ENTITY_TYPES = {
    "financial_metrics": [
        "NET PROFIT", "Operating Profit", "Total Income", "Interest Income", 
        "Other Income", "Total Expenses", "Interest Expenses", "Operating Expenses", "Provisions"
    ],
    "business_segments": [
        "RETAIL BANKING SEGMENT", "WHOLESALE BANKING SEGMENT", "TREASURY SEGMENT",
        "LIFE INSURANCE SEGMENT", "OTHERS SEGMENT"
    ],
    "financial_ratios": [
        "Basic EPS", "Diluted EPS", "Net Margin", "Operating Margin", "Cost Ratio"
    ],
    "balance_sheet_items": [
        "Advances", "Investments", "Customer Deposits", "Total Assets", "Total Equity",
        "Cash & RBI Balances", "Borrowings", "Share Capital", "Reserves & Surplus"
    ]
}

SUPPORTED_QUARTERS = ["Q1_FY2024", "Q2_FY2024", "Q3_FY2024", "Q4_FY2024"]

CHUNK_TYPES = [
    "profitability_analysis", "balance_sheet_analysis", 
    "financial_ratios", "segment_analysis"
]

SUPPORTED_MODELS = {
    "gemini-2.0-flash": {"rate_limit": 4.0, "max_tokens": 8192},
    "gemini-1.5-pro": {"rate_limit": 2.0, "max_tokens": 8192},
    "gpt-3.5-turbo": {"rate_limit": 1.0, "max_tokens": 8192},
    "llama3.1:8b": {"rate_limit": 0.5, "max_tokens": 4096},
    "groq-llama": {"rate_limit": 0.5, "max_tokens": 8192}
}

# -------------------------
# Helper Functions for Data Validation (removed custom validation, using Pydantic)
# -------------------------

def validate_quarter(quarter: str) -> bool:
    """Validate quarter format"""
    return quarter in SUPPORTED_QUARTERS

def validate_chunk_type(chunk_type: str) -> bool:
    """Validate chunk type"""
    return chunk_type in CHUNK_TYPES

# -------------------------
# MCP Server Setup
# -------------------------
server = FastMCP("graph_rag_tool")

# -------------------------
# Health Check & Info Tools
# -------------------------

@server.tool("health_check", description="Check Graph RAG server health and dependencies")
def mcp_health_check():
    """Comprehensive health check for all Graph RAG dependencies"""
    config = get_config()
    health_status = {
        "server": "healthy",
        "timestamp": time.time(),
        "config": {
            "neo4j_uri": config.neo4j_uri,
            "default_model": config.default_model,
            "supported_models": list(SUPPORTED_MODELS.keys())
        },
        "dependencies": {
            "pydantic": True,  # We imported it successfully
            "aiohttp": True,   # We imported it successfully
        },
        "llm_providers": {},
        "issues": []
    }
    
    # Check configuration
    config_issues = config.validate()
    if config_issues:
        health_status["issues"].extend(config_issues)
        health_status["server"] = "warning"
    
    # Check LLM provider availability
    try:
        # Gemini
        if config.gemini_api_key:
            try:
                import google.generativeai
                health_status["dependencies"]["google_generativeai"] = True
                health_status["llm_providers"]["gemini"] = "available"
            except ImportError:
                health_status["dependencies"]["google_generativeai"] = False
                health_status["llm_providers"]["gemini"] = "missing_package"
                health_status["issues"].append("google-generativeai package not installed")
        else:
            health_status["llm_providers"]["gemini"] = "no_api_key"
        
        # OpenAI
        if config.openai_api_key:
            try:
                import openai
                health_status["dependencies"]["openai"] = True
                health_status["llm_providers"]["openai"] = "available"
            except ImportError:
                health_status["dependencies"]["openai"] = False
                health_status["llm_providers"]["openai"] = "missing_package"
                health_status["issues"].append("openai package not installed")
        else:
            health_status["llm_providers"]["openai"] = "no_api_key"
        
        # Groq/Llama
        if config.groq_api_key:
            health_status["llm_providers"]["groq"] = "available"
        else:
            health_status["llm_providers"]["groq"] = "no_api_key"
        
        # Local Llama (always available)
        health_status["llm_providers"]["llama_local"] = "available_if_ollama_running"
        
    except Exception as e:
        health_status["issues"].append(f"LLM provider check failed: {str(e)}")
        health_status["server"] = "warning"
    
    # Check default model availability
    default_api_key = config.get_api_key_for_model(config.default_model)
    if not default_api_key and "llama" not in config.default_model.lower():
        health_status["issues"].append(f"Default model {config.default_model} has no API key configured")
        health_status["server"] = "warning"
    
    # TODO: Add Neo4j health check in next step
    health_status["dependencies"]["neo4j"] = "not_checked_yet"
    
    return health_status

@server.tool("get_server_info", description="Get Graph RAG server information and capabilities")
def mcp_get_server_info():
    """Return comprehensive server information"""
    config = get_config()
    
    return {
        "server_name": "Graph RAG Financial Analysis Server",
        "version": "1.0.0",
        "description": "Standalone Graph RAG for financial quarterly analysis",
        "port": config.port,
        "capabilities": {
            "entity_extraction": True,
            "knowledge_graph": True,
            "natural_language_queries": True,
            "multi_llm_support": True,
            "financial_analysis": True
        },
        "supported_features": {
            "entity_types": list(FINANCIAL_ENTITY_TYPES.keys()),
            "quarters": SUPPORTED_QUARTERS,
            "models": list(SUPPORTED_MODELS.keys())
        },
        "endpoints": {
            "health": "health_check",
            "info": "get_server_info", 
            "extract": "extract_financial_entities",
            "build": "build_financial_graph",
            "query": "query_financial_graph",
            "stats": "get_graph_stats"
        }
    }



'''
@server.tool("test_llm_providers", description="Test LLM provider initialization and connectivity")
def mcp_test_llm_providers():
    """Test all available LLM providers with current configuration"""
    config = get_config()
    results = {}
    
    # Test Gemini Provider
    try:
        if config.gemini_api_key:
            gemini = GeminiProvider("gemini-2.0-flash", config.gemini_api_key)
            results["gemini"] = {
                "status": "initialized", 
                "model": gemini.model_name,
                "rate_limit": gemini.rate_limit,
                "api_key_configured": True
            }
        else:
            results["gemini"] = {
                "status": "no_api_key",
                "api_key_configured": False
            }
    except Exception as e:
        results["gemini"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Test GPT Provider  
    try:
        if config.openai_api_key:
            gpt = GPTProvider("gpt-3.5-turbo", config.openai_api_key)
            results["gpt"] = {
                "status": "initialized",
                "model": gpt.model_name, 
                "rate_limit": gpt.rate_limit,
                "api_key_configured": True
            }
        else:
            results["gpt"] = {
                "status": "no_api_key",
                "api_key_configured": False
            }
    except Exception as e:
        results["gpt"] = {
            "status": "error", 
            "error": str(e)
        }
    
    # Test Llama Provider (Groq)
    try:
        if config.groq_api_key:
            llama_groq = LlamaProvider("llama3.1:8b", config.groq_api_key)
            results["llama_groq"] = {
                "status": "initialized",
                "model": llama_groq.model_name,
                "use_groq": llama_groq.use_groq,
                "rate_limit": llama_groq.rate_limit,
                "api_key_configured": True
            }
        else:
            results["llama_groq"] = {
                "status": "no_api_key", 
                "api_key_configured": False
            }
    except Exception as e:
        results["llama_groq"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Test Llama Provider (Local)
    try:
        llama_local = LlamaProvider("llama3.1:8b", None)  # No API key = local
        results["llama_local"] = {
            "status": "initialized",
            "model": llama_local.model_name,
            "use_groq": llama_local.use_groq,
            "base_url": llama_local.base_url,
            "note": "Requires Ollama running locally"
        }
    except Exception as e:
        results["llama_local"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Summary
    initialized_count = sum(1 for r in results.values() if r.get("status") == "initialized")
    
    return {
        "status": "complete",
        "message": f"LLM provider test complete - {initialized_count}/{len(results)} providers initialized",
        "default_model": config.default_model,
        "providers": results,
        "summary": {
            "total_tested": len(results),
            "initialized": initialized_count,
            "missing_api_keys": sum(1 for r in results.values() if r.get("status") == "no_api_key"),
            "errors": sum(1 for r in results.values() if r.get("status") == "error")
        }
    }
'''

@server.tool("test_llm_generation", description="Test LLM content generation with a simple prompt")
async def mcp_test_generation(model_name: str = None, test_prompt: str = "What is 2+2?"):
    """Test actual content generation with specified model"""
    config = get_config()
    model_name = model_name or config.default_model
    
    try:
        # Get appropriate API key
        api_key = config.get_api_key_for_model(model_name)
        
        # Create provider based on model
        if "gemini" in model_name.lower():
            if not api_key:
                return {"status": "error", "message": "Gemini API key not configured"}
            provider = GeminiProvider(model_name, api_key)
        elif "gpt" in model_name.lower():
            if not api_key:
                return {"status": "error", "message": "OpenAI API key not configured"}
            provider = GPTProvider(model_name, api_key)
        elif "llama" in model_name.lower():
            provider = LlamaProvider(model_name, api_key)  # Can work with or without API key
        else:
            return {"status": "error", "message": f"Unsupported model: {model_name}"}
        
        # Test generation
        start_time = time.time()
        response = await provider.generate_content(test_prompt)
        generation_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": "Content generation successful",
            "model_used": model_name,
            "test_prompt": test_prompt,
            "response": response,
            "generation_time_seconds": round(generation_time, 2),
            "response_length": len(response),
            "provider_type": type(provider).__name__
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Content generation failed: {str(e)}",
            "model_attempted": model_name,
            "error_details": str(e)
        }


@server.tool("extract_financial_entities", description="Extract financial entities from quarterly report text")
async def mcp_extract_entities(text: str, period: str = None, model: str = None):
    """Extract structured financial entities from text using your exact EntityExtractor"""
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

@server.tool("extract_and_save_to_graph", description="Complete pipeline: extract entities from text and save to graph")
def mcp_extract_and_save(text: str, period: str = None, model: str = None, dataset_id: str = "mcp_pipeline"):
    """Complete pipeline: extract financial entities and save directly to Neo4j graph"""
    try:
        # Step 1: Extract entities directly (don't call the MCP tool)
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
        
        import concurrent.futures
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
        
        import concurrent.futures
        
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
@server.tool("build_financial_graph", description="Build knowledge graph from financial chunks")
def mcp_build_graph(chunks: List[Dict], dataset_id: str = "financial_data", clear_existing: bool = False):
    """Build graph from chunks - fixed async handling"""
    try:
        config = get_config()
        neo4j_service = Neo4jService(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
        
        if clear_existing:
            neo4j_service.clear_data(dataset_id)
        
        processed = failed = total_entities = 0
        failed_chunks = []
        
        for chunk_data in chunks:
            try:
                # Convert to FinancialChunk
                chunk = FinancialChunk.model_validate(chunk_data)
                
                # Extract entities using the working method from extract_and_save_to_graph
                extractor = EntityExtractor(config.default_model, config.get_api_key_for_model(config.default_model))
                
                import concurrent.futures
                async def extract_async():
                    return await extractor.extract(chunk)
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, extract_async())
                    entities = future.result(timeout=config.entity_extraction_timeout)
                
                if not entities.quarter:
                    failed += 1
                    failed_chunks.append(chunk.id)
                    continue
                
                # Save entities
                neo4j_service.save_entities(entities, chunk.id, dataset_id)
                processed += 1
                total_entities += len(entities.financial_metrics) + len(entities.business_segments) + len(entities.financial_ratios) + len(entities.balance_sheet_items)
                
            except Exception as e:
                logger.error(f"Failed chunk {chunk_data.get('id', 'unknown')}: {e}")
                failed += 1
                failed_chunks.append(chunk_data.get('id', 'unknown'))
        
        stats = neo4j_service.get_stats()
        neo4j_service.close()
        
        return {
            "status": "success",
            "message": "Graph built successfully",
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
@server.tool("traverse_entity_relationships", description="Find related entities across the knowledge graph")
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

# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    config = get_config()
    
    # Validate configuration on startup
    issues = config.validate()
    if issues:
        logger.warning("Configuration issues detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("✅ Configuration validated successfully")
    
    logger.info(f"🚀 Starting Graph RAG MCP Server")
    logger.info(f"📊 Default model: {config.default_model}")
    logger.info(f"🔗 Neo4j URI: {config.neo4j_uri}")
    logger.info(f"🌐 Server: {config.host}:{config.port}")
    logger.info(f"📖 Supported entity types: {len(FINANCIAL_ENTITY_TYPES)} types")
    logger.info(f"🤖 Supported models: {len(SUPPORTED_MODELS)} models")
    
    try:
        server.run(transport="streamable-http", host=config.host, port=config.port)
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)