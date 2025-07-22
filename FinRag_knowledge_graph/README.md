# Financial Knowledge Graph System

This system extracts financial data from quarterly reports and builds a **Neo4j-based knowledge graph**, allowing **natural language queries** using LLMs like Gemini, GPT-3.5, and LLaMA 3.1.



## Components

### Entity Service (`PORT 8001`)

* Extracts financial entities from raw text using LLMs
* Consumed internally by Graph Service
* Not required to be run separately unless debugging

### Graph Service (`PORT 8002`)

* Builds and queries the knowledge graph
* Exposes REST APIs for building and querying

###  Neo4j Database

* Stores all extracted entities and relationships

###  LLM Providers

* **Gemini** (via Gemini API)
* **GPT-3.5** (via OpenAI)
* **LLaMA 3.1 8B** (via Ollama/Groq)

---

## Getting Started

### 1 Environment Setup

* Ensure the following API keys are set in your environment:

```bash
export GEMINI_API_KEY=your_key_here
export OPENAI_API_KEY=optional_key
export GROQ_API_KEY=optional_key
```

* Ensure Neo4j is running and accessible.

---

### 2️ Running the Services

####  Start Entity Extraction Service (optional unless debugging)

```bash
python main_entity.py --model gemini-2.0-flash  # or llama3.1:8b
```

####  Start Graph Service

```bash
python main_graph.py --model gemini-2.0-flash  # default model
```

or

```bash
python main_graph.py --model llama3.1:8b
```

>  You can extract entities with one LLM and query the graph with another.

---

##  Data Ingestion

To add new financial reports (e.g., ICICI quarterly reports):

###  POST to `/api/v1/build`

**URL:** `http://localhost:8002/api/v1/build`
**Headers:** `Content-Type: application/json`
**Body:**

```json
{
  "chunks": [
    {
      "id": "test_chunk_1",
      "period": "Q1_FY2024",
      "type": "profitability_analysis",
      "size": 500,
      "text": "ICICI Bank reported a net profit of ₹10,636 crores for Q1 FY2024..."
    },
    {
      "id": "test_chunk_2",
      "period": "Q2_FY2024",
      "type": "segment_analysis",
      "size": 400,
      "text": "In Q2 FY2024, ICICI Bank's wholesale banking segment..."
    }
  ],
  "dataset_id": "icici_test",
  "clear_existing": true
}
```

---

##  Asking Questions

###  POST to `/api/v1/query`

**URL:** `http://localhost:8002/api/v1/query`
**Body:**

```json
{
  "question": "What was ICICI Bank's net profit in Q1 FY2024?",
  "limit": 10
}
```

---

##  Data Model

###  Financial Metrics

* **Net Profit**, **Operating Profit**, **Total Income**
* **Interest Income**, **Other Income**, **Total Expenses**
* **Interest Expenses**, **Operating Expenses**, **Provisions**

###  Financial Ratios

* **Cost Ratio**, **Net Margin**, **Operating Margin**
* **Basic EPS**, **Diluted EPS**

###  Business Segments

* **Retail Banking Segment**, **Wholesale Banking Segment**
* **Treasury Segment**, **Life Insurance Segment**, **Others Segment**

###  Balance Sheet Items

* **Advances**, **Investments**, **Customer Deposits**
* **Total Assets**, **Total Equity**

###  Available Quarters

* `Q1_FY2024`, `Q2_FY2024`, `Q3_FY2024`, `Q4_FY2024`

---

##  API Reference

###  Graph Service (Port 8002)

| Endpoint         | Method | Description                    |
| ---------------- | ------ | ------------------------------ |
| `/api/v1/build`  | POST   | Build graph from chunked data  |
| `/api/v1/query`  | POST   | Ask natural language questions |
| `/api/v1/stats`  | GET    | View graph stats & metadata    |
| `/api/v1/health` | GET    | Service health check           |
| `/ping`          | GET    | Ping test                      |

---

###  Entity Service (Port 8001)

| Endpoint                | Method | Description                  |
| ----------------------- | ------ | ---------------------------- |
| `/api/v1/extract`       | POST   | Extract from single chunk    |
| `/api/v1/extract/batch` | POST   | Extract from multiple chunks |
| `/api/v1/health`        | GET    | Service health check         |
| `/ping`                 | GET    | Ping test                    |


##  Dev Notes

* Can switch between LLMs for extraction vs. querying
* Ollama runs LLaMA locally — ensure it’s installed and serving
* Milvus vector DB used for storing and retrieving preprocessed chunks

---
