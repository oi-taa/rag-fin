import google.generativeai as genai
from neo4j import GraphDatabase
import json
import re
from sentence_transformers import SentenceTransformer
genai.configure(api_key="AIzaSyD6YknQ0kPpFO5Xtb175B47mTzYcJPJtDM")
import time

class SimpleRateLimiter:
    def __init__(self):
        self.last_request = 0
    
    def wait(self):
        # 4 seconds between requests = 15 per minute max
        elapsed = time.time() - self.last_request
        if elapsed < 4:
            time.sleep(4 - elapsed)
        self.last_request = time.time()

class FinancialHybridRAG:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, milvus_collection_name):
        self.rate_limiter = SimpleRateLimiter()
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Connect to existing Milvus
        from pymilvus import Collection, connections
        connections.connect("default", host="localhost", port="19530")
        self.vector_store = Collection(milvus_collection_name)
        self.vector_store.load()
        
        self.chunk_to_entities = {}
        self._create_neo4j_schema()
    
    def load_chunks_from_milvus(self):
        """Load all chunks from Milvus for KG construction"""
        
        # Get all chunk IDs first
        all_chunks = self.vector_store.query(
            expr="",  # Empty expression gets all
            limit = 1000,
            output_fields=["id", "text", "period", "chunk_type", "statement_type", "primary_value"]
        )
        
        print(f"Loaded {len(all_chunks)} chunks from Milvus")
        return all_chunks
    def build_knowledge_graph_from_milvus(self):
        """Build KG directly from Milvus data"""
        
        print("Loading chunks from Milvus...")
        chunks = self.load_chunks_from_milvus()
        
        print("Building Knowledge Graph from Milvus chunks...")
        self.build_knowledge_graph(chunks)
    
    def _create_neo4j_schema(self):
        """Create constraints and basic schema"""
        with self.neo4j_driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (org:Organization) REQUIRE org.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (q:Quarter) REQUIRE q.name IS UNIQUE") 
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Metric) REQUIRE m.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Segment) REQUIRE s.name IS UNIQUE")
    

    def _get_default_structure(self):
        """Return default structure when parsing fails"""
        return {
            "quarter": None,
            "financial_metrics": [],
            "business_segments": [],
            "financial_ratios": [],
            "balance_sheet_items": []
        }

    def extract_structured_entities(self, chunk):
        """Extract structured financial data using Gemini with function calling"""
        try:
            self.rate_limiter.wait() 
            model = genai.GenerativeModel(
                'gemini-2.0-flash',
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                    # No response_schema parameter
                )
            )

            prompt = f"""
            Extract ALL financial data from this ICICI Bank quarterly report chunk:
            {chunk['text']}

            Return a JSON object that strictly follows this schema:

            {{
                "quarter": "string (e.g., Q1_FY2024)",
                "financial_metrics": [
                    {{
                        "name": "string (exact name from text)",
                        "value": "number (simple format, max 2 decimals)",
                        "growth_yoy": "number (optional, percentage as decimal)",
                        "unit": "string (e.g., crore, percentage)"
                    }}
                ],
                "business_segments": [
                    {{
                        "name": "string (exact segment name)", 
                        "revenue": "number (in crores)",
                        "margin": "number (percentage as decimal)",
                        "percentage_of_total": "number (optional)"
                    }}
                ],
                "financial_ratios": [
                    {{
                        "name": "string (ratio name)",
                        "value": "number (simple format)",
                        "growth_yoy": "number (optional)",
                        "unit": "string (e.g., per share, percentage)"
                    }}
                ],
                "balance_sheet_items": [
                    {{
                        "name": "string (item name)",
                        "value": "number (in crores)", 
                        "percentage_of_total": "number (optional)",
                        "unit": "string (usually crore)"
                    }}
                ]
            }}

            Categorization rules:
            - financial_metrics: profits, income, expenses, operating metrics
            - business_segments: segment revenue, margins, performance  
            - financial_ratios: EPS, margins, efficiency ratios
            - balance_sheet_items: assets, deposits, capital, funding items

            Number formatting rules:
            - Convert ₹1,124,875 crore to 1124875 (remove commas, keep as crores)
            - Convert +43.3% YoY to 43.3 (just the number)
            - Use simple numbers: 15.22 not 15.220000000001
            - Maximum 2 decimal places
            - No scientific notation
            - Use exact names from text

            Return ONLY the JSON object, no explanations.
            """
            
            
            # DEBUG: Print what Gemini actually returned
        
            response = model.generate_content(prompt)
            response_text = response.text
            response_text = re.sub(r'(\d+)\.0{20,}', r'\1.0', response_text)
            response_text = re.sub(r'(\d+\.\d{1,2})\d{20,}', r'\1', response_text)

            print(f"=== CLEANED RESPONSE ===")
            print(f"Cleaned text (first 500 chars): {response_text[:500]}")
            
            parsed_response = json.loads(response_text)
            return parsed_response
        
        except json.JSONDecodeError as e:
            print(f"JSON decode error for chunk {chunk.get('id', 'unknown')}: {e}")
            print(f"Raw response: {response.text[:1000]}...")  # Show first 1000 chars
            return self._get_default_structure()
    
        except Exception as e:
            print(f"Unexpected error for chunk {chunk.get('id', 'unknown')}: {e}")
            return self._get_default_structure()


        
    def build_knowledge_graph(self, chunks):
        """Main function: Build KG from your chunks with bidirectional linking"""
        
        print("Building Knowledge Graph from financial chunks...")
        
        for chunk in chunks:
            print(f"Processing: {chunk['id']}")
            
            # Extract structured entities
            structured_data = self.extract_structured_entities(chunk)
            
            # Create Neo4j nodes and relationships
            self._create_neo4j_entities(structured_data, chunk['id'])
            
            # Update vector store metadata for linking
            #self._update_vector_metadata(chunk, structured_data)
            
            # Track for bidirectional lookup
            self.chunk_to_entities[chunk['id']] = structured_data
        
        print("Knowledge Graph construction complete!")
    
    def _create_neo4j_entities(self, data, chunk_id):
        """Create Neo4j nodes and relationships from structured data"""
        
        with self.neo4j_driver.session() as session:
            quarter = data.get('quarter')
            if not quarter:
                return
            
            # Create quarter node
            session.run("""
                MERGE (org:Organization {name: "ICICI_Bank_Limited"})
                MERGE (q:Quarter {name: $quarter})
                SET q.source_chunks = COALESCE(q.source_chunks, []) + $chunk_id
                MERGE (org)-[:HAS_QUARTER]->(q)
            """, quarter=quarter, chunk_id=chunk_id)
            
            # Create financial metric relationships
            for metric in data.get('financial_metrics', []):  # ✅ Fixed key
                session.run("""
                    MATCH (q:Quarter {name: $quarter})
                    MERGE (m:Metric {name: $metric_name})
                    CREATE (q)-[:HAS_METRIC {
                        value: $value,
                        growth_yoy: $growth,
                        unit: $unit,
                        source_chunk: $chunk_id
                    }]->(m)
                """, quarter=quarter, metric_name=metric['name'],
                    value=metric['value'], growth=metric.get('growth_yoy'),
                    unit=metric.get('unit'), chunk_id=chunk_id)
            
            # Create segment relationships
            for segment in data.get('business_segments', []):  # ✅ Fixed key
                session.run("""
                    MATCH (q:Quarter {name: $quarter})
                    MERGE (s:Segment {name: $segment_name})
                    CREATE (q)-[:HAS_SEGMENT_PERFORMANCE {
                        revenue: $revenue,
                        margin: $margin,
                        percentage_of_total: $percentage,
                        source_chunk: $chunk_id
                    }]->(s)
                """, quarter=quarter, segment_name=segment['name'],
                    revenue=segment.get('revenue'), margin=segment.get('margin'),
                    percentage=segment.get('percentage_of_total'), chunk_id=chunk_id)
            
            # ✅ ADD: Create financial ratio relationships
            for ratio in data.get('financial_ratios', []):
                session.run("""
                    MATCH (q:Quarter {name: $quarter})
                    MERGE (r:Ratio {name: $ratio_name})
                    CREATE (q)-[:HAS_RATIO {
                        value: $value,
                        growth_yoy: $growth,
                        unit: $unit,
                        source_chunk: $chunk_id
                    }]->(r)
                """, quarter=quarter, ratio_name=ratio['name'],
                    value=ratio['value'], growth=ratio.get('growth_yoy'),
                    unit=ratio.get('unit'), chunk_id=chunk_id)
            
            # ✅ ADD: Create balance sheet relationships  
            for item in data.get('balance_sheet_items', []):
                session.run("""
                    MATCH (q:Quarter {name: $quarter})
                    MERGE (b:BalanceSheetItem {name: $item_name})
                    CREATE (q)-[:HAS_BALANCE_SHEET_ITEM {
                        value: $value,
                        percentage_of_total: $percentage,
                        unit: $unit,
                        source_chunk: $chunk_id
                    }]->(b)
                """, quarter=quarter, item_name=item['name'],
                    value=item['value'], percentage=item.get('percentage_of_total'),
                    unit=item.get('unit'), chunk_id=chunk_id)

    def hybrid_query_simple(self, question):
        """Hybrid query without metadata updates"""
    
        # Vector search - FIXED
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = embedding_model.encode([question])
    
        vector_results = self.vector_store.search(
            query_embedding,
            "embedding",  # your vector field name
            {"metric_type": "COSINE"},
            limit=1000,
            output_fields=["id", "text", "period", "chunk_type"]
        )
    
        # Extract chunks from Milvus results
        vector_chunks = []
        if vector_results and len(vector_results) > 0:
            for hit in vector_results[0]:
                vector_chunks.append({
                    'id': hit.entity.get('id'),
                    'text': hit.entity.get('text'),
                    'period': hit.entity.get('period'),
                    'chunk_type': hit.entity.get('chunk_type'),
                    'score': hit.score
                })
    
        # Graph search - FIXED
        graph_entities = self.graph_search(question)
    
        # Get chunks from graph (using source_chunk property) - FIXED
        graph_chunk_ids = [e.get('source_chunk') for e in graph_entities if e.get('source_chunk')]
        
        # FIXED: Use query instead of get_by_ids
        graph_chunks = []
        if graph_chunk_ids:
            try:
                # Convert list to proper format for Milvus query
                ids_str = str(graph_chunk_ids).replace("'", '"')
                
                chunk_results = self.vector_store.query(
                    expr=f"id in {ids_str}",
                    output_fields=["id", "text", "period", "chunk_type"]
                )
                
                for chunk in chunk_results:
                    graph_chunks.append({
                        'id': chunk.get('id'),
                        'text': chunk.get('text'),
                        'period': chunk.get('period'),
                        'chunk_type': chunk.get('chunk_type'),
                        'score': 1.0  # High score for graph matches
                    })
                    
            except Exception as e:
                print(f"Error retrieving graph chunks: {e}")
                graph_chunks = []
    
        # Merge results and remove duplicates
        seen_ids = set()
        all_chunks = []
        
        # Add vector chunks
        for chunk in vector_chunks:
            if chunk['id'] not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk['id'])
        
        # Add graph chunks (avoid duplicates)
        for chunk in graph_chunks:
            if chunk['id'] not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk['id'])
    
        return all_chunks
    
    
    def graph_search(self, question):
        """Focused graph search - returns only relevant combinations"""
        
        question_entities = self._extract_entities_from_question(question)
        print(f"Using entities: {question_entities}")
        
        # Group entities by type for smarter querying
        quarters = [e['name'] for e in question_entities if e['type'] == 'Quarter']
        segments = [e['name'] for e in question_entities if e['type'] == 'Segment'] 
        metrics = [e['name'] for e in question_entities if e['type'] == 'Metric']
        
        print(f"Found: {len(quarters)} quarters, {len(segments)} segments, {len(metrics)} metrics")
        print(f"🔍 Strategy conditions:")
        print(f"   segments and len(quarters) > 1: {bool(segments and len(quarters) > 1)}")
        print(f"   metrics and len(quarters) > 1: {bool(metrics and len(quarters) > 1)}")
        print(f"   len(quarters) == 1: {bool(len(quarters) == 1)}")
        print(f"   segments and not quarters: {bool(segments and not quarters)}")
        print(f"   metrics and not quarters: {bool(metrics and not quarters)}")
        
        results = []
        with self.neo4j_driver.session() as session:
            
            # Strategy 1: Specific segment across multiple quarters
            if segments and len(quarters) > 1:
                print(f"📊 Strategy: {segments[0]} across {len(quarters)} quarters")
                for segment in segments:
                    result = session.run("""
                        MATCH (q:Quarter)-[r:HAS_SEGMENT_PERFORMANCE]->(s:Segment {name: $segment})
                        WHERE q.name IN $quarters
                        RETURN q.name as quarter, s.name as segment_name,
                            r.revenue as revenue, r.margin as margin,
                            r.source_chunk as source_chunk
                        ORDER BY q.name
                    """, segment=segment, quarters=quarters)
                    results.extend(result.data())
            
            # Strategy 2: Specific metric across multiple quarters  
            elif metrics and len(quarters) > 1:
                print(f"📊 Strategy: {metrics[0]} across {len(quarters)} quarters")
                for metric in metrics:
                    result = session.run("""
                        MATCH (q:Quarter)-[r:HAS_METRIC]->(m:Metric {name: $metric})
                        WHERE q.name IN $quarters
                        RETURN q.name as quarter, m.name as metric_name,
                            r.value as value, r.growth_yoy as growth,
                            r.unit as unit, r.source_chunk as source_chunk
                        ORDER BY q.name
                    """, metric=metric, quarters=quarters)
                    results.extend(result.data())
            
            # Strategy 3: Single quarter with specific segment/metric
            elif len(quarters) == 1:
                quarter = quarters[0]
                print(f"📊 Strategy: Deep dive into {quarter}")
                
                # Get specific segments if mentioned
                if segments:
                    for segment in segments:
                        result = session.run("""
                            MATCH (q:Quarter {name: $quarter})-[r:HAS_SEGMENT_PERFORMANCE]->(s:Segment {name: $segment})
                            RETURN q.name as quarter, s.name as segment_name,
                                r.revenue as revenue, r.margin as margin,
                                r.source_chunk as source_chunk
                        """, quarter=quarter, segment=segment)
                        results.extend(result.data())
                
                # Get specific metrics if mentioned
                if metrics:
                    for metric in metrics:
                        result = session.run("""
                            MATCH (q:Quarter {name: $quarter})-[r:HAS_METRIC]->(m:Metric {name: $metric})
                            RETURN q.name as quarter, m.name as metric_name,
                                r.value as value, r.growth_yoy as growth,
                                r.unit as unit, r.source_chunk as source_chunk
                        """, quarter=quarter, metric=metric)
                        results.extend(result.data())
                
                # If no specific entities, get top metrics only
                if not segments and not metrics:
                    result = session.run("""
                        MATCH (q:Quarter {name: $quarter})-[r:HAS_METRIC]->(m:Metric)
                        WHERE m.name IN ['NET PROFIT', 'Operating Profit', 'Total']
                        RETURN q.name as quarter, m.name as metric_name,
                            r.value as value, r.growth_yoy as growth,
                            r.unit as unit, r.source_chunk as source_chunk
                    """, quarter=quarter)
                    results.extend(result.data())
            
            # Strategy 4: Segment comparison (when segment mentioned but quarters aren't specific)
            elif segments and not quarters:
                print(f"📊 Strategy: {segments[0]} across all quarters")
                for segment in segments:
                    result = session.run("""
                        MATCH (q:Quarter)-[r:HAS_SEGMENT_PERFORMANCE]->(s:Segment {name: $segment})
                        RETURN q.name as quarter, s.name as segment_name,
                            r.revenue as revenue, r.margin as margin,
                            r.source_chunk as source_chunk
                        ORDER BY q.name
                    """, segment=segment)
                    results.extend(result.data())
            
            # Strategy 5: Metric trend (when metric mentioned but quarters aren't specific) 
            elif metrics and not quarters:
                print(f"📊 Strategy: {metrics[0]} trend analysis")
                for metric in metrics:
                    result = session.run("""
                        MATCH (q:Quarter)-[r:HAS_METRIC]->(m:Metric {name: $metric})
                        RETURN q.name as quarter, m.name as metric_name,
                            r.value as value, r.growth_yoy as growth,
                            r.unit as unit, r.source_chunk as source_chunk
                        ORDER BY q.name
                    """, metric=metric)
                    results.extend(result.data())
            
            # Strategy 6: Question-specific patterns
            else:
                print("📊 Strategy: Pattern-based search")
                
                # "Compare retail banking performance" pattern
                if 'retail' in question.lower() and any(word in question.lower() for word in ['compare', 'performance', 'across']):
                    result = session.run("""
                        MATCH (q:Quarter)-[r:HAS_SEGMENT_PERFORMANCE]->(s:Segment {name: "RETAIL BANKING SEGMENT"})
                        RETURN q.name as quarter, s.name as segment_name,
                            r.revenue as revenue, r.margin as margin,
                            r.source_chunk as source_chunk
                        ORDER BY q.name
                    """)
                    results.extend(result.data())
        
        print(f"📋 Graph search returned {len(results)} focused results")
        
        # Safety limit
        if len(results) > 30:
            print(f"⚠️  Limiting results from {len(results)} to 30 most relevant")
            results = results[:30]
        
        return results

    def _extract_entities_from_question(self, question):
        """Extract entities from question using Gemini with comprehensive name mapping"""
        
        try:
            self.rate_limiter.wait() 
            model = genai.GenerativeModel('gemini-2.0-flash')
            
            prompt = f"""
            You are a financial data analyst. Analyze this question step-by-step to extract entities.

            Question: "{question}"

            Think through this systematically:

            STEP 1: QUESTION TYPE ANALYSIS
            - Is this asking about a specific quarter or multiple quarters?
            - Is this comparing segments or asking about one segment?
            - Is this asking for a specific metric or general performance?
            - Keywords that indicate comparison: "which", "compare", "best", "drove growth", "performed better"

            STEP 2: ENTITY IDENTIFICATION
            
            Time Periods:
            - Look for: Q1, Q2, Q3, Q4, "first quarter", "second quarter", etc.
            - Map to: Q1_FY2024, Q2_FY2024, Q3_FY2024, Q4_FY2024
            - If question asks "which business drove growth" across segments → need the specific quarter mentioned

            Business Segments:
            - Look for: "retail", "wholesale", "treasury", "corporate", "insurance", "life insurance", "others"
            - Map to: RETAIL BANKING SEGMENT, WHOLESALE BANKING SEGMENT, TREASURY SEGMENT, LIFE INSURANCE SEGMENT, OTHERS SEGMENT
            - If question asks "which segment" or "which business" → include ALL segments for comparison

            Financial Metrics:
            - "revenue" → Total
            - "profit" → NET PROFIT  
            - "margin" → Net Margin
            - "performance" → context-dependent (revenue for segments, profit for overall)
            - "profitability" → Net Margin

            STEP 3: COMPARATIVE LOGIC
            - If question contains "which", "compare", "best", "drove growth" → this needs comparison
            - For segment comparison → include all 5 segments
            - For quarter comparison → include all relevant quarters

            STEP 4: OUTPUT FORMAT
            Based on your analysis, extract entities in this JSON format:

            {{
                "reasoning": "Brief explanation of your step-by-step analysis",
                "entities": [
                    {{"name": "exact_entity_name", "type": "Quarter|Segment|Metric|Ratio|BalanceSheetItem"}}
                ]
            }}

            EXAMPLES:

            Question: "What was retail banking revenue in Q2?"
            Reasoning: "Single quarter (Q2), single segment (retail banking), specific metric (revenue)"
            Entities: [
                {{"name": "Q2_FY2024", "type": "Quarter"}},
                {{"name": "RETAIL BANKING SEGMENT", "type": "Segment"}}, 
                {{"name": "Total", "type": "Metric"}}
            ]

            Question: "Which business segment drove growth in Q3?"
            Reasoning: "Comparative question ('which') asking about segment performance in Q3. Need all segments for comparison."
            Entities: [
                {{"name": "Q3_FY2024", "type": "Quarter"}},
                {{"name": "RETAIL BANKING SEGMENT", "type": "Segment"}},
                {{"name": "WHOLESALE BANKING SEGMENT", "type": "Segment"}},
                {{"name": "TREASURY SEGMENT", "type": "Segment"}},
                {{"name": "LIFE INSURANCE SEGMENT", "type": "Segment"}},
                {{"name": "OTHERS SEGMENT", "type": "Segment"}}
            ]

            Question: "How did treasury margins evolve across quarters?"
            Reasoning: "Single segment (treasury) across all quarters ('evolve across'). Evolution requires multiple time points."
            Entities: [
                {{"name": "TREASURY SEGMENT", "type": "Segment"}},
                {{"name": "Net Margin", "type": "Metric"}},
                {{"name": "Q1_FY2024", "type": "Quarter"}},
                {{"name": "Q2_FY2024", "type": "Quarter"}},
                {{"name": "Q3_FY2024", "type": "Quarter"}},
                {{"name": "Q4_FY2024", "type": "Quarter"}}
            ]

            Now analyze: "{question}"

            Return ONLY the JSON with reasoning and entities. No markdown formatting.
            """
            
            response = model.generate_content(prompt)
            print(response)
            
            # Get text from correct location
            if hasattr(response, 'candidates') and response.candidates:
                raw_text = response.candidates[0].content.parts[0].text
            else:
                raw_text = response.text
            
            # Clean markdown
            cleaned = re.sub(r'```json\n?', '', raw_text)
            cleaned = re.sub(r'\n?```', '', cleaned).strip()
            
            # Parse and return entities
            parsed = json.loads(cleaned)
            entities = parsed.get('entities', [])
            
            # Entity name mapping based on actual Neo4j nodes
            '''name_mappings = {
                # Quarters
                'q1_fy2024': 'Q1_FY2024',
                'q2_fy2024': 'Q2_FY2024', 
                'q3_fy2024': 'Q3_FY2024',
                'q4_fy2024': 'Q4_FY2024',
                'q1': 'Q1_FY2024',
                'quarter 1': 'Q1_FY2024',
                'first quarter': 'Q1_FY2024',
                'q2': 'Q2_FY2024',
                'quarter 2': 'Q2_FY2024',
                'second quarter': 'Q2_FY2024',
                'q3': 'Q3_FY2024',
                'quarter 3': 'Q3_FY2024',
                'third quarter': 'Q3_FY2024',
                'q4': 'Q4_FY2024',
                'quarter 4': 'Q4_FY2024',
                'fourth quarter': 'Q4_FY2024',
                
                # Profit/Income Metrics
                'net profit': 'NET PROFIT',
                'profit': 'NET PROFIT',
                'net income': 'NET PROFIT',
                'operating profit': 'Operating Profit',
                'operating income': 'Operating Profit',
                
                # Income Types
                'interest income': 'Interest Income',
                'other income': 'Other Income',
                'total income': 'Total',
                'revenue': 'Total',
                
                # Expenses
                'interest': 'Interest',
                'interest expense': 'Interest',
                'operating': 'Operating',
                'operating expenses': 'Operating',
                'provisions': 'Provisions',
                
                # Balance Sheet Items
                'cash': 'Cash & RBI Balances',
                'cash and rbi': 'Cash & RBI Balances',
                'rbi balances': 'Cash & RBI Balances',
                'deposits': 'Customer Deposits',
                'customer deposits': 'Customer Deposits',
                'borrowings': 'Borrowings',
                'share capital': 'Share Capital',
                'capital': 'Share Capital',
                'reserves': 'Reserves & Surplus',
                'surplus': 'Reserves & Surplus',
                'reserves and surplus': 'Reserves & Surplus',
                'total equity': 'Total Equity',
                'equity': 'Total Equity',
                'advances': 'Advances',
                'loans': 'Advances',
                'investments': 'Investments',
                
                # EPS
                'basic eps': 'Basic EPS',
                'diluted eps': 'Diluted EPS',
                'eps': 'Basic EPS',
                'earnings per share': 'Basic EPS',
                
                # Segments
                'retail banking': 'RETAIL BANKING SEGMENT',
                'retail': 'RETAIL BANKING SEGMENT',
                'treasury': 'TREASURY SEGMENT',
                'treasury segment': 'TREASURY SEGMENT',
                'wholesale banking': 'WHOLESALE BANKING SEGMENT',
                'wholesale': 'WHOLESALE BANKING SEGMENT',
                'corporate banking': 'WHOLESALE BANKING SEGMENT',
                'corporate': 'WHOLESALE BANKING SEGMENT',
                'life insurance': 'LIFE INSURANCE SEGMENT',
                'insurance': 'LIFE INSURANCE SEGMENT',
                'others': 'OTHERS SEGMENT',
                'other segments': 'OTHERS SEGMENT',
                
                # Ratios and Margins
                'net margin': 'Net Margin',
                'operating margin': 'Operating Margin',
                'cost ratio': 'Cost Ratio',
                'deposit to funding ratio': 'Deposit-to-Funding Ratio',
                'deposit funding ratio': 'Deposit-to-Funding Ratio',
                'margin': 'Net Margin',  # Generic margin
                'margins': 'Net Margin',
                
                # Segment Results
                'segment result': 'Segment Result',
                'segment results': 'Segment Result',
                
                # Company
                'icici': 'ICICI_Bank_Limited',
                'icici bank': 'ICICI_Bank_Limited',
            }
            
            # Fix entity names to match Neo4j format
            fixed_entities = []
            for entity in entities:
                name = entity['name'].lower().strip()
                entity_type = entity['type']
                
                print(f"🔍 Looking up: '{name}' (original: '{entity['name']}')")
                # Check if we have a direct mapping
                if name in name_mappings:
                    mapped_name = name_mappings[name]
                    fixed_entities.append({'name': mapped_name, 'type': entity_type})
                else:
                    # Try partial matching for complex terms
                    mapped = False
                    for key, value in name_mappings.items():
                        if key in name or any(word in name for word in key.split()):
                            fixed_entities.append({'name': value, 'type': entity_type})
                            mapped = True
                            break
                    
                    # If no mapping found, keep original but format it
                    if not mapped:
                        # Convert to title case and replace spaces with underscores for consistency
                        formatted_name = '_'.join(word.capitalize() for word in name.split())
                        fixed_entities.append({'name': formatted_name, 'type': entity_type})
            
            # Remove duplicates
            seen = set()
            unique_entities = []
            for entity in fixed_entities:
                entity_key = f"{entity['name']}_{entity['type']}"
                if entity_key not in seen:
                    seen.add(entity_key)
                    unique_entities.append(entity)
                    '''
            # Replace your entire mapping section with:
            fixed_entities = []
            for entity in entities:
                # Since Gemini now returns correct format, just use as-is
                fixed_entities.append({
                    'name': entity['name'], 
                    'type': entity['type']
                })
            
            print(f"✅ Extracted entities: {fixed_entities}")
            return fixed_entities
            
        except Exception as e:
            print(f"❌ Unexpected error in entity extraction: {e}")
            print(f"📥 Error type: {type(e)}")
            import traceback
            print(f"📥 Full traceback: {traceback.format_exc()}")
            return []
def main():
    # Initialize system
    rag = FinancialHybridRAG(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        milvus_collection_name="fin_chunks"
    )
    
    rag.build_knowledge_graph_from_milvus()
    
    result = rag.hybrid_query_simple("How did ICICI's net profit change from Q1 to Q4 FY2024?")
    print("Hybrid retrieval result:", result)

if __name__ == "__main__":
    main()