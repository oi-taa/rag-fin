import json
import os
import glob
from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

# Initialize the sentence transformer model for generating embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Milvus vector database running on localhost
connections.connect("default", host="localhost", port="19530")

# Define the collection schema for storing financial data chunks
fields = [
    FieldSchema("id", DataType.VARCHAR, max_length=100, is_primary=True),
    FieldSchema("text", DataType.VARCHAR, max_length=4000),  # Increased limit for longer text
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=384),
    FieldSchema("period", DataType.VARCHAR, max_length=20),
    FieldSchema("chunk_type", DataType.VARCHAR, max_length=30),
    FieldSchema("statement_type", DataType.VARCHAR, max_length=30),
    FieldSchema("primary_value", DataType.DOUBLE),
]

# Drop existing collection if it exists and create a new one
if utility.has_collection("fin_chunks"):
    utility.drop_collection("fin_chunks")

collection = Collection("fin_chunks", CollectionSchema(fields, "Financial complete context chunks"))
collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}})

def create_complete_context_chunks(quarterly_data, period):
    """
    Create comprehensive context chunks from quarterly financial data.
    Focuses only on consolidated financial statements for consistency.
    
    Args:
        quarterly_data: List of JSON data objects containing financial statements
        period: String representing the financial period (e.g., "Q1_FY2024")
    
    Returns:
        List of chunk dictionaries containing processed financial data
    """
    chunks = []
    
    # Identify consolidated financial statements from the quarterly data
    consolidated_fin = None
    consolidated_segmental = None
    consolidated_balance = None
    
    # Parse through all data files to find consolidated statements
    for data in quarterly_data:
        if data.get("reportType") == "CONSOLIDATED FINANCIAL RESULTS":
            consolidated_fin = data
        elif data.get("reportType") == "CONSOLIDATED SEGMENTAL RESULTS":
            consolidated_segmental = data
            print(f"    Found CONSOLIDATED SEGMENTAL RESULTS file for {period}")
        elif "consolidatedSegmentalResults" in data:
            consolidated_segmental = data
            print(f"    Found consolidatedSegmentalResults in data for {period}")
        elif "consolidatedBalanceSheet" in data:
            consolidated_balance = data
        elif "segmentalResults" in data and not consolidated_segmental:  # Fallback option
            consolidated_segmental = data
            print(f"    Found segmentalResults as fallback for {period}")
    
    # Exit early if no consolidated financial data is available
    if not consolidated_fin:
        print(f"  No consolidated financial data found for {period}")
        return chunks
    
    company = consolidated_fin.get("company", "ICICI Bank Limited")
    
    # Determine the correct period keys based on the quarter
    current_period = None
    prev_year_period = None
    
    # Map quarter to corresponding month keys in the JSON data
    if "Q1" in period:
        current_period = "june2023" if "2024" in period else "june2022"
        prev_year_period = "june2022" if "2024" in period else "june2021"
    elif "Q2" in period:
        current_period = "september2023" if "2024" in period else "september2022"
        prev_year_period = "september2022" if "2024" in period else "september2021"
    elif "Q3" in period:
        current_period = "december2023" if "2024" in period else "december2022"
        prev_year_period = "december2022" if "2024" in period else "december2021"
    elif "Q4" in period:
        current_period = "march2024" if "2024" in period else "march2023"
        prev_year_period = "march2023" if "2024" in period else "march2022"
    
    # CHUNK 1: PROFITABILITY ANALYSIS
    # Extract income statement data and calculate profitability metrics
    if "consolidatedResults" in consolidated_fin:
        results = consolidated_fin["consolidatedResults"]
        
        if current_period and "income" in results and "expenses" in results and "profitAndLoss" in results:
            income = results["income"]
            expenses = results["expenses"]
            pnl = results["profitAndLoss"]
            
            # Extract current period financial metrics
            total_income = income["totalIncome"].get(current_period, 0)
            interest_income = income["interestEarned"].get(current_period, 0)
            other_income = income["otherIncome"].get(current_period, 0)
            
            total_expenses = expenses["totalExpenditure"].get(current_period, 0)
            interest_expenses = expenses["interestExpended"].get(current_period, 0)
            operating_expenses = expenses["operatingExpenses"].get(current_period, 0)
            
            operating_profit = pnl["operatingProfit"].get(current_period, 0)
            net_profit = pnl["netProfitForThePeriod"].get(current_period, 0)
            provisions = pnl["provisions"].get(current_period, 0)
            
            # Extract previous year data for year-over-year comparison
            prev_total_income = income["totalIncome"].get(prev_year_period, 0)
            prev_net_profit = pnl["netProfitForThePeriod"].get(prev_year_period, 0)
            
            # Calculate key profitability ratios
            net_margin = (net_profit / total_income * 100) if total_income else 0
            operating_margin = (operating_profit / total_income * 100) if total_income else 0
            cost_to_income = (total_expenses / total_income * 100) if total_income else 0
            
            # Calculate growth rates
            income_growth = ((total_income - prev_total_income) / prev_total_income * 100) if prev_total_income else 0
            profit_growth = ((net_profit - prev_net_profit) / prev_net_profit * 100) if prev_net_profit else 0
            
            # Create comprehensive profitability analysis text
            chunk_text = f"{company} {period} NET PROFIT PROFITABILITY ANALYSIS:\n\n"
            chunk_text += f"NET PROFIT: ₹{net_profit:,.0f} crore"
            if prev_net_profit:
                chunk_text += f" ({profit_growth:+.1f}% YoY growth)"
            chunk_text += f"\nOperating Profit: ₹{operating_profit:,.0f} crore"
            chunk_text += f"\nNet Margin: {net_margin:.1f}% | Operating Margin: {operating_margin:.1f}%\n\n"
            
            chunk_text += f"INCOME: Total ₹{total_income:,.0f} crore"
            if prev_total_income:
                chunk_text += f" ({income_growth:+.1f}% YoY)"
            chunk_text += f"\nInterest Income: ₹{interest_income:,.0f} crore ({interest_income/total_income*100:.1f}%)"
            chunk_text += f"\nOther Income: ₹{other_income:,.0f} crore ({other_income/total_income*100:.1f}%)\n\n"
            
            chunk_text += f"EXPENSES: Total ₹{total_expenses:,.0f} crore"
            chunk_text += f"\nInterest: ₹{interest_expenses:,.0f} crore | Operating: ₹{operating_expenses:,.0f} crore"
            chunk_text += f"\nProvisions: ₹{provisions:,.0f} crore | Cost Ratio: {cost_to_income:.1f}%"
            
            chunks.append({
                "id": f"icici_{period.lower()}_profitability_analysis",
                "text": chunk_text,
                "period": period,
                "chunk_type": "profitability_analysis",
                "statement_type": "consolidated",
                "primary_value": net_profit
            })
    
    # CHUNK 2: BALANCE SHEET ANALYSIS
    # Extract balance sheet data and analyze financial position
    if consolidated_balance and "consolidatedBalanceSheet" in consolidated_balance:
        bs = consolidated_balance["consolidatedBalanceSheet"]
        if bs and "assets" in bs and "capitalAndLiabilities" in bs:
            assets = bs["assets"]
            liabilities = bs["capitalAndLiabilities"]
            
            # Extract key balance sheet components
            total_assets = assets["totalAssets"].get(current_period, 0)
            advances = assets["advances"].get(current_period, 0)
            investments = assets["investments"].get(current_period, 0)
            cash_rbi = assets["cashAndBalancesWithRBI"].get(current_period, 0)
            
            deposits = liabilities["deposits"].get(current_period, 0)
            borrowings = liabilities["borrowings"].get(current_period, 0)
            capital = liabilities["capital"].get(current_period, 0)
            reserves = liabilities["reservesAndSurplus"].get(current_period, 0)
            
            # Calculate asset composition and funding ratios
            advances_ratio = (advances / total_assets * 100) if total_assets else 0
            investments_ratio = (investments / total_assets * 100) if total_assets else 0
            casa_ratio = (deposits / (deposits + borrowings) * 100) if (deposits + borrowings) else 0
            
            # Create balance sheet analysis text
            chunk_text = f"{company} {period} Balance Sheet Analysis:\n\n"
            chunk_text += f"ASSET COMPOSITION (Total: ₹{total_assets:,.0f} crore):\n"
            chunk_text += f"• Advances: ₹{advances:,.0f} crore ({advances_ratio:.1f}% of total assets)\n"
            chunk_text += f"• Investments: ₹{investments:,.0f} crore ({investments_ratio:.1f}% of total assets)\n"
            chunk_text += f"• Cash & RBI Balances: ₹{cash_rbi:,.0f} crore\n\n"
            
            chunk_text += f"FUNDING STRUCTURE:\n"
            chunk_text += f"• Customer Deposits: ₹{deposits:,.0f} crore\n"
            chunk_text += f"• Borrowings: ₹{borrowings:,.0f} crore\n"
            chunk_text += f"• Deposit-to-Funding Ratio: {casa_ratio:.1f}%\n\n"
            
            chunk_text += f"CAPITAL POSITION:\n"
            chunk_text += f"• Share Capital: ₹{capital:,.0f} crore\n"
            chunk_text += f"• Reserves & Surplus: ₹{reserves:,.0f} crore\n"
            chunk_text += f"• Total Equity: ₹{capital + reserves:,.0f} crore"
            
            chunks.append({
                "id": f"icici_{period.lower()}_balance_sheet_health",
                "text": chunk_text,
                "period": period,
                "chunk_type": "balance_sheet_analysis",
                "statement_type": "consolidated",
                "primary_value": total_assets
            })
    
    # CHUNK 3: KEY FINANCIAL RATIOS AND METRICS
    # Extract and analyze key performance indicators
    if consolidated_fin and "consolidatedResults" in consolidated_fin:
        results = consolidated_fin["consolidatedResults"]
        
        if "ratios" in results:
            ratios = results["ratios"]
            
            chunk_text = f"{company} {period} Key Financial Ratios & Metrics:\n\n"
            
            # Extract earnings per share data
            if "earningsPerShare" in ratios:
                eps = ratios["earningsPerShare"]
                basic_eps = eps["basic"].get(current_period, 0)
                diluted_eps = eps["diluted"].get(current_period, 0)
                prev_basic_eps = eps["basic"].get(prev_year_period, 0)
                
                # Calculate EPS growth
                eps_growth = ((basic_eps - prev_basic_eps) / prev_basic_eps * 100) if prev_basic_eps else 0
                
                chunk_text += f"EARNINGS METRICS:\n"
                chunk_text += f"• Basic EPS: ₹{basic_eps:.2f} per share"
                if prev_basic_eps:
                    chunk_text += f" ({eps_growth:+.1f}% YoY)"
                chunk_text += f"\n• Diluted EPS: ₹{diluted_eps:.2f} per share\n\n"
        
        # Only create chunk if there's substantial content
        if len(chunk_text) > 100:
            chunks.append({
                "id": f"icici_{period.lower()}_key_ratios",
                "text": chunk_text,
                "period": period,
                "chunk_type": "financial_ratios",
                "statement_type": "consolidated",
                "primary_value": basic_eps if 'basic_eps' in locals() else 0
            })
    
    # CHUNK 4: SEGMENT PERFORMANCE ANALYSIS
    # Extract and analyze business segment performance data
    segment_data_found = False
    seg_results = None
    
    # Search for segmental data in various possible formats
    if consolidated_segmental:
        print(f"    Processing segmental data for {period}")
        if "consolidatedSegmentalResults" in consolidated_segmental:
            seg_results = consolidated_segmental["consolidatedSegmentalResults"]
            segment_data_found = True
            print(f"    Using consolidatedSegmentalResults structure")
        elif "segmentalResults" in consolidated_segmental:
            seg_results = consolidated_segmental["segmentalResults"]
            segment_data_found = True
            print(f"    Using segmentalResults structure")
        else:
            print(f"    No recognized segmental structure in {list(consolidated_segmental.keys())}")
    else:
        print(f"    No segmental data file found for {period}")
    
    # Process segment data if found
    if segment_data_found and seg_results:
        
        # Identify the correct data structure for segment information
        if "segmentRevenue" in seg_results and "segmentResults" in seg_results:
            revenue_data = seg_results["segmentRevenue"]
            profit_data = seg_results["segmentResults"]
        elif "segmentRevenue" in seg_results and "segmentalResults" in seg_results:
            revenue_data = seg_results["segmentRevenue"]  
            profit_data = seg_results["segmentalResults"]
        else:
            print(f"    Segmental data keys: {list(seg_results.keys())}")
            print(f"    No compatible segment structure found for {period}")
            return chunks
        
        chunk_text = f"{company} {period} Retail Banking & Business Segment Performance:\n\n"
        
        # Define business segments to analyze
        segments = {
            "retailBanking": "Retail Banking",
            "wholesaleBanking": "Wholesale Banking",
            "treasury": "Treasury",
            "lifeInsurance": "Life Insurance",
            "others": "Others"
        }
        
        total_revenue = 0
        segment_details = []
        
        # Extract revenue and profit data for each segment
        for seg_key, seg_name in segments.items():
            if seg_key in revenue_data and current_period in revenue_data[seg_key]:
                revenue = revenue_data[seg_key][current_period]
                profit = profit_data[seg_key].get(current_period, 0) if seg_key in profit_data else 0
                
                total_revenue += revenue
                margin = (profit / revenue * 100) if revenue else 0
                
                segment_details.append({
                    "name": seg_name,
                    "revenue": revenue,
                    "profit": profit,
                    "margin": margin
                })
        
        # Sort segments by revenue size for better presentation
        segment_details.sort(key=lambda x: x["revenue"], reverse=True)
        
        # Create detailed segment analysis text
        for seg in segment_details:
            revenue_pct = (seg["revenue"] / total_revenue * 100) if total_revenue else 0
            chunk_text += f"{seg['name'].upper()} SEGMENT:\n"
            chunk_text += f"• Revenue: ₹{seg['revenue']:,.0f} crore ({revenue_pct:.1f}%)\n"
            chunk_text += f"• Segment Result: ₹{seg['profit']:,.0f} crore\n"
            chunk_text += f"• Margin: {seg['margin']:.1f}%\n\n"
        
        chunk_text += f"TOTAL SEGMENT REVENUE: ₹{total_revenue:,.0f} crore"
        
        chunks.append({
            "id": f"icici_{period.lower()}_segment_performance",
            "text": chunk_text,
            "period": period,
            "chunk_type": "segment_analysis",
            "statement_type": "consolidated",
            "primary_value": total_revenue
        })
        print(f"    Successfully created segment analysis chunk for {period}")
    else:
        print(f"    No consolidated segmental data found for {period}")

    return chunks

# Main processing logic
all_chunks = []
data_folder = "extract_data"

# Define quarters to process and their corresponding period names
quarters = ["q1_2023", "q2_2023", "q3_2023", "q4_2023"]
period_mapping = {
    "q1_2023": "Q1_FY2024",
    "q2_2023": "Q2_FY2024", 
    "q3_2023": "Q3_FY2024",
    "q4_2023": "Q4_FY2024"
}

# Process each quarter's financial data
for quarter in quarters:
    print(f"\nProcessing {quarter}...")
    quarter_folder = os.path.join(data_folder, f"icici_{quarter}")
    
    if os.path.exists(quarter_folder):
        # Load all JSON files from the quarter folder
        quarterly_data = []
        json_files = glob.glob(os.path.join(quarter_folder, "*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    quarterly_data.append(data)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        # Generate comprehensive context chunks for this quarter
        period = period_mapping[quarter]
        chunks = create_complete_context_chunks(quarterly_data, period)
        all_chunks.extend(chunks)
        
        print(f"Generated {len(chunks)} complete context chunks for {period}")
        for chunk in chunks:
            print(f"  - {chunk['chunk_type']}")

print(f"\nTotal chunks created: {len(all_chunks)}")

# Store processed chunks in Milvus vector database
if all_chunks:
    # Extract text content and generate embeddings
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = model.encode(texts).tolist()
    
    # Prepare data for insertion into Milvus
    data = [
        [chunk["id"] for chunk in all_chunks],
        texts,
        embeddings,
        [chunk["period"] for chunk in all_chunks],
        [chunk["chunk_type"] for chunk in all_chunks],
        [chunk["statement_type"] for chunk in all_chunks],
        [chunk["primary_value"] for chunk in all_chunks]
    ]
    
    # Insert data into the collection and load it for searching
    collection.insert(data)
    collection.flush()
    collection.load()
    print(f"\nInserted {len(all_chunks)} chunks into Milvus")

def search_financial_query(query, top_k=3):
    """
    Search function to retrieve relevant financial information based on queries.
    
    Args:
        query: Natural language query about financial data
        top_k: Number of top results to return
    """
    # Generate embedding for the search query
    query_embedding = model.encode([query])
    
    # Perform similarity search in the vector database
    results = collection.search(
        query_embedding, 
        "embedding", 
        {"metric_type": "COSINE"}, 
        top_k,
        output_fields=["text", "period", "chunk_type", "statement_type", "primary_value"]
    )
    
    # Display search results with relevance scores
    print(f"\nQuery: '{query}'\n" + "="*50)
    for i, result in enumerate(results[0], 1):
        print(f"{i}. Score: {result.score:.3f} | Type: {result.entity.chunk_type} | Period: {result.entity.period}")
        print(f"Text preview: {result.entity.text[:200]}...")
        print("-" * 50)

# Test the financial RAG system with sample queries
print("\n" + "="*70)
print("TESTING COMPLETE CONTEXT FINANCIAL RAG")
print("="*70)

search_financial_query("What was ICICI's Q1 net profit and profitability?")
search_financial_query("How did retail banking perform in Q2?")
search_financial_query("What are the key financial ratios for Q3?")