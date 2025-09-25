# Graph RAG MCP Server

A standalone MCP server that builds knowledge graphs from financial documents and enables natural language querying using LLM-generated Cypher queries.

## Overview

Converts unstructured financial text and structured JSON data into a Neo4j knowledge graph, then allows querying via natural language that gets translated to Cypher queries.

**Architecture:**
```
Financial Data → Entity Extraction → Neo4j Graph → Natural Language Queries → Cypher → Results
```

## Prerequisites

- Neo4j running on `localhost:7687`
- API keys for LLM providers (Gemini/OpenAI/Groq)
- Python dependencies: `neo4j`, `google-generativeai`, `openai`, `fastmcp`

## Environment Setup

Create `.env` file:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

GEMINI_API_KEY=your_key_here
DEFAULT_MODEL=gemini-2.0-flash
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd graph_rag_mcp

# Install dependencies
pip install neo4j google-generativeai openai fastmcp

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials
```

## Running the Server

```bash
cd graph_rag_mcp
python main.py
```

Server runs on port 9008. Use MCP Inspector: `npx @modelcontextprotocol/inspector`

## Available Tools

### System Health & Info

#### `health_check`
**Purpose:** Verify Neo4j connection and LLM availability  
**Parameters:** None  
**Sample:**
```json
{
  "tool": "health_check",
  "result": "All systems operational ✓"
}
```

#### `get_server_info`
**Purpose:** Show system capabilities and supported entity types  
**Parameters:** None  
**Sample:**
```json
{
  "tool": "get_server_info",
  "result": {
    "supported_quarters": ["Q1_FY2024", "Q2_FY2024", "Q3_FY2024", "Q4_FY2024"],
    "entity_types": ["metrics", "segments", "ratios", "balance_sheet"],
    "models": ["gemini-2.0-flash", "gpt-4", "groq-llama"]
  }
}
```

### Entity Extraction & Graph Building

#### `extract_financial_entities`
**Purpose:** Extract structured entities from financial text  
**Parameters:**
- `text` (string): Financial text to analyze
- `period` (string): Quarter period (e.g., "Q1_FY2024")

**Sample:**
```json
{
  "tool": "extract_financial_entities",
  "text": "ICICI Bank Q1_FY2024 NET PROFIT: ₹10,636 crore (+44% YoY)",
  "period": "Q1_FY2024",
  "result": {
    "metrics": [
      {
        "name": "NET PROFIT",
        "value": "10636",
        "unit": "crore",
        "growth": "+44% YoY"
      }
    ],
    "ratios": [],
    "segments": []
  }
}
```

#### `build_financial_graph`
**Purpose:** Build knowledge graph from multiple chunks (supports both text and structured JSON)  
**Parameters:**
- `chunks` (array): List of financial data chunks
- `dataset_id` (string): Identifier for the dataset
- `clear_existing` (boolean): Whether to clear existing graph data

**Sample:**
```json
{
  "tool": "build_financial_graph",
  "chunks": [
    {
      "id": "icici_q1",
      "period": "Q1_FY2024",
      "type": "profitability_analysis", 
      "size": 200,
      "text": "ICICI Bank Q1_FY2024 NET PROFIT: ₹10,636 crore"
    },
    {
      "company": "Axis Bank",
      "financialResults": {
        "income": {
          "totalIncome": {
            "march2024Annual": "139922.79"
          }
        }
      }
    }
  ],
  "dataset_id": "multi_bank_data",
  "clear_existing": true,
  "result": {
    "organizations_created": 2,
    "entities_created": 8,
    "relationships_created": 15,
    "processing_time": "2.3s"
  }
}
```

#### `get_graph_stats`
**Purpose:** Show graph statistics (node counts, relationships)  
**Parameters:** None  
**Sample:**
```json
{
  "tool": "get_graph_stats",
  "result": {
    "organizations": 2,
    "quarters": 4,
    "metrics": 15,
    "ratios": 8,
    "segments": 12,
    "relationships": 45
  }
}
```

### Graph Querying

#### `query_financial_graph`
**Purpose:** Natural language queries converted to Cypher  
**Parameters:**
- `question` (string): Natural language question
- `limit` (integer, optional): Maximum results to return

**Samples:**

Simple metric lookup:
```json
{
  "tool": "query_financial_graph",
  "question": "What was ICICI Bank's net profit in Q1 FY2024?",
  "result": {
    "answer": "₹10,636 crore",
    "cypher_used": "MATCH (o:Organization {name: 'ICICI Bank'})-[:HAS_QUARTER]->(q:Quarter {period: 'Q1_FY2024'})-[:HAS_METRIC]->(m:Metric {name: 'NET PROFIT'}) RETURN m.value",
    "execution_time": "0.12s"
  }
}
```

Comparative analysis:
```json
{
  "tool": "query_financial_graph",
  "question": "Compare net profit between ICICI and Axis Bank",
  "result": {
    "answer": "ICICI Q1: ₹10,636 crore, Axis Q4: ₹32,110 crore",
    "details": [
      {"bank": "ICICI Bank", "quarter": "Q1_FY2024", "net_profit": "10636 crore"},
      {"bank": "Axis Bank", "quarter": "Q4_FY2024", "net_profit": "32110 crore"}
    ]
  }
}
```

Segment analysis:
```json
{
  "tool": "query_financial_graph",
  "question": "Show all segments for Q1 FY2024",
  "result": {
    "segments": [
      {"name": "Retail Banking", "revenue": "31057 crore", "margin": "13.5%"},
      {"name": "Treasury", "revenue": "26306 crore", "margin": "16.6%"},
      {"name": "Wholesale Banking", "revenue": "16960 crore", "margin": "24.1%"}
    ]
  }
}
```

#### `generate_cypher_query`
**Purpose:** Generate Cypher without executing (for debugging)  
**Parameters:**
- `question` (string): Natural language question

**Sample:**
```json
{
  "tool": "generate_cypher_query",
  "question": "Show profit trend across quarters",
  "result": {
    "cypher": "MATCH (q:Quarter)-[:HAS_METRIC]->(m:Metric {name: 'NET PROFIT'}) RETURN q.period, m.value ORDER BY q.period",
    "explanation": "This query finds all quarters with NET PROFIT metrics and returns them in chronological order"
  }
}
```

#### `execute_fallback_query`
**Purpose:** Safe predefined queries when main query fails  
**Parameters:**
- `query_type` (string): Type of fallback query ("overview", "metrics", "segments")
- `limit` (integer, optional): Maximum results to return

**Sample:**
```json
{
  "tool": "execute_fallback_query",
  "query_type": "overview",
  "limit": 5,
  "result": {
    "description": "Top 5 metrics across all quarters",
    "data": [
      {"metric": "NET PROFIT", "quarter": "Q4_FY2024", "value": "11672 crore"},
      {"metric": "Total Income", "quarter": "Q4_FY2024", "value": "67082 crore"}
    ]
  }
}
```

## Supported Data Formats

### Text Format (LLM Extraction)
For unstructured financial documents that need entity extraction:

```json
{
  "id": "chunk_id",
  "period": "Q1_FY2024",
  "type": "profitability_analysis",
  "size": 300,
  "text": "ICICI Bank Q1_FY2024 NET PROFIT: ₹10,636 crore (+44% YoY growth). Operating Profit reached ₹14,186 crore with a net margin of 18.8%..."
}
```

### Structured Format (Direct Mapping)
For pre-structured financial data:

```json
{
  "company": "Axis Bank",
  "period": "Q4_FY2024",
  "financialResults": {
    "income": {
      "totalIncome": {"march2024Annual": "139922.79"},
      "interestIncome": {"march2024Annual": "89156.45"}
    },
    "ratios": {
      "RoA (%)": {"march2024": "2.45"},
      "RoE (%)": {"march2024": "18.23"}
    },
    "segments": {
      "retailBanking": {
        "revenue": "45678.90",
        "margin": "15.2%"
      }
    }
  }
}
```

## Entity Types Supported

### Financial Metrics
- **Profit & Loss:** NET PROFIT, Operating Profit, Total Income, Interest Income, Other Income
- **Expenses:** Operating Expenses, Interest Expenses, Provisions
- **Growth:** YoY Growth percentages, QoQ comparisons

### Business Segments
- **Banking Segments:** Retail Banking, Treasury, Wholesale Banking, Corporate Banking
- **Insurance:** Life Insurance, General Insurance
- **Others:** Investment Banking, Digital Services

### Financial Ratios
- **Profitability:** Net Margin, Operating Margin, ROA, ROE
- **Per Share:** Basic EPS, Diluted EPS, Book Value per Share
- **Efficiency:** Cost Ratio, Cost-to-Income Ratio

### Balance Sheet Items
- **Assets:** Advances, Investments, Cash & RBI Balances, Total Assets
- **Liabilities:** Customer Deposits, Borrowings, Other Liabilities
- **Equity:** Share Capital, Reserves & Surplus, Total Equity

## Graph Schema

```
Organization (ICICI Bank, Axis Bank)
    ↓ [HAS_QUARTER]
Quarter (Q1_FY2024, Q2_FY2024, Q3_FY2024, Q4_FY2024)
    ↓ [HAS_METRIC] [HAS_SEGMENT_PERFORMANCE] [HAS_RATIO] [HAS_BALANCE_SHEET_ITEM]
Financial Entities (Metrics, Segments, Ratios, Balance Sheet Items)
```

**Node Types:**
- **Organization:** Bank/Financial institution
- **Quarter:** Reporting period
- **Metric:** Financial metrics (NET PROFIT, Total Income, etc.)
- **Segment:** Business segment performance
- **Ratio:** Financial ratios and percentages
- **BalanceSheetItem:** Assets, liabilities, equity items

**Relationship Types:**
- **HAS_QUARTER:** Organization → Quarter
- **HAS_METRIC:** Quarter → Metric
- **HAS_SEGMENT_PERFORMANCE:** Quarter → Segment
- **HAS_RATIO:** Quarter → Ratio
- **HAS_BALANCE_SHEET_ITEM:** Quarter → BalanceSheetItem

## Query Capabilities

### ✅ Supported Queries
- **Simple lookups:** "What was ICICI's net profit in Q1?"
- **Trend analysis:** "Show profit growth across quarters"
- **Cross-company comparison:** "Compare ICICI vs Axis Bank performance"
- **Multi-hop queries:** "Which segments had the highest margins?"
- **Aggregations:** "Total revenue across all segments"
- **Filtering:** "Show only metrics above ₹10,000 crore"

### ❌ Limitations
- **Causal reasoning:** "Why did performance improve?" (requires Vector RAG)
- **External context:** "How does this compare to industry averages?"
- **Predictive queries:** "What will next quarter's profit be?"
- **Unstructured analysis:** "Analyze management commentary"

## Architecture Notes

This Graph RAG server is designed to work in a **distributed MCP architecture** alongside:

- **Vector RAG MCP server:** Semantic search and document understanding
- **Query Routing MCP server:** Intelligent query distribution and intent classification
- **Attention Layer:** Result fusion and ranking from multiple RAG systems

### Standalone Usage
Works independently for structured financial data queries where relationships and patterns are key.

### Distributed Usage
Handles graph-based queries in hybrid RAG system:
- **Graph RAG:** Structured relationships, trends, comparisons
- **Vector RAG:** Semantic understanding, context, explanations
- **Router:** Determines which system(s) to query
- **Attention:** Combines and ranks results

## Error Handling

The server includes robust error handling:

- **Connection errors:** Automatic Neo4j reconnection attempts
- **LLM failures:** Fallback to simpler extraction methods
- **Query failures:** Fallback queries for common patterns
- **Validation errors:** Schema validation for input data

## Performance Considerations

- **Batch processing:** Supports bulk graph building for large datasets
- **Query optimization:** Cypher query optimization and indexing
- **Caching:** Results caching for repeated queries
- **Memory management:** Efficient handling of large financial datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Create an issue in the repository
- Check the troubleshooting guide
- Review the MCP documentation