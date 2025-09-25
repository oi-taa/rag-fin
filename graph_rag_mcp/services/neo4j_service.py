 
"""
Neo4j database operations for Graph RAG MCP Server
EXACT copy from graph/neo4j_service.py - preserving all functionality
"""

from neo4j import GraphDatabase
from typing import List, Dict, Any
import logging
from models.financial_models import ExtractedEntities

logger = logging.getLogger(__name__)

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
    
    def save_entities(self, entities: ExtractedEntities, chunk_id: str, dataset_id: str, company_name: str = "ICICI Bank"):
        """Save all entity types with properties stored ON NODES"""
        
        with self.driver.session() as session:
            quarter = entities.quarter
            if not quarter:
                return
            
            # Create quarter and organization with dynamic company name
            session.run("""
                MERGE (org:Organization {name: $company_name})
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
            """, company_name=company_name, quarter=quarter, dataset_id=dataset_id, chunk_id=chunk_id)
            
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