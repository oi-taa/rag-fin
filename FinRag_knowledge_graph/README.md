┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   LLM Query      │───▶│   Neo4j Graph   │
│  (Natural Lang) │    │   Generator      │    │   Database      │
└─────────────────┘    └──────────────────┘    └─────────────────┘

Components:

Entity Service (Port 8001): Extracts financial data from quarterly reports into JSON- don't need to run its only for extraction(from input chunks)and is imported into graph service
Graph Service (Port 8002): Builds knowledge graph and handles queries
Neo4j Database: Stores financial entities and relationships
LLM Providers: Gemini, GPT-3.5-turbo, Llama 3.1:8b 

Commands to run:

python main_entity.py --model gemini-2.0-flash (default)
python main_graph.py --model gemini-2.0-flash (default)
python main_graph.py --model llama3.1:8b
python main_entity.py --model llama3.1:8b
Ollama installed locally 

API Keys  - set in env

Gemini API Key: Required for Gemini models
OpenAI API Key: Optional for GPT models
Groq API Key: Optional for free Llama via Groq(tested llama not groq)

Data Ingestion - Adding New Quarterly Reports
The system will automatically:

1. Extract financial entities using the selected LLM
2. Store entities in Neo4j graph database
3. Create relationships between quarters and financial data
4. Return processing results
5. can also build with one llm the graph and query with another 

Using the Build Method
To add new ICICI quarterly reports to the system:
get chunks from milvus - format as given in chunks.json
POST- http://localhost:8002/api/v1/build -- for building the graph - send chunks
Content-Type: application/json
sample - 
{
  "chunks": [
    {
      "id": "test_chunk_1",
      "period": "Q1_FY2024",
      "type": "profitability_analysis",
      "size": 500,
      "text": "ICICI Bank reported a net profit of ₹10,636 crores for Q1 FY2024, representing a growth of 44% year-on-year. The bank's retail banking segment contributed ₹31,057 crores in revenue with a net margin of 13.5%. Basic EPS stood at ₹15.22 per share."
    },
    {
      "id": "test_chunk_2", 
      "period": "Q2_FY2024",
      "type": "segment_analysis",
      "size": 400,
      "text": "In Q2 FY2024, ICICI Bank's wholesale banking segment generated revenue of ₹18,245 crores with an operating margin of 8.7%. Treasury operations contributed ₹5,832 crores. Customer deposits reached ₹1,245,890 crores."
    }
  ],
  "dataset_id": "icici_test",
  "clear_existing": true
}


questions- 
POST - http://localhost:8002/api/v1/query
Content-Type: application/json
{
  "question": "What was ICICI Bank's net profit in Q1 FY2024?",
  "limit": 10
}
![alt text](image.png)

Data Model
Available Entities
Financial Metrics:

NET PROFIT, Operating Profit, Total Income
Interest Income, Other Income, Total Expenses
Interest Expenses, Operating Expenses, Provisions

Financial Ratios:

Cost Ratio, Net Margin, Operating Margin
Basic EPS, Diluted EPS

Business Segments:

RETAIL BANKING SEGMENT, WHOLESALE BANKING SEGMENT
TREASURY SEGMENT, LIFE INSURANCE SEGMENT, OTHERS SEGMENT

Balance Sheet Items:

Advances, Investments, Customer Deposits
Total Assets, Total Equity

Quarters Available:

Q1_FY2024, Q2_FY2024, Q3_FY2024, Q4_FY2024

Graph Service Endpoints (Port 8002)
GET /api/v1/stats
Returns graph statistics and available data.
GET /api/v1/health
Service health check.
GET /ping
Simple ping endpoint.

Entity Service Endpoints (Port 8001)
POST /api/v1/extract
Extract entities from a single financial text chunk.
POST /api/v1/extract/batch
Extract entities from multiple chunks.
GET /api/v1/health
Service health check.
GET /ping
Simple ping endpoint.
