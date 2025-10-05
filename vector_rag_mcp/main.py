"""
Vector RAG MCP Server - Port 9006
"""

import os
import sys
import logging
import time
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load env FIRST
load_dotenv()

# Set port BEFORE any imports
os.environ.setdefault("PORT", "9006")

from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections
import google.generativeai as genai

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBhPk2fEF7tDNlaWZxv5oECp26-lBCiuLk")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "fin_chunks")

class VectorRAG:
    def __init__(self, gemini_api_key: str, collection_name: str = "fin_chunks"):
        """Initialize RAG system"""
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection = Collection(collection_name)
        self.collection.load()
        logger.info(f"Connected to Milvus collection: {collection_name}")
    
    def search(self, query: str, top_k: int = 3):
        """Search vector store"""
        query_embedding = self.similarity_model.encode([query])
        results = self.collection.search(
            query_embedding,
            "embedding",
            {"metric_type": "COSINE"},
            top_k,
            output_fields=["text", "period", "chunk_type", "statement_type", "primary_value"]
        )
        
        contexts = []
        for i, result in enumerate(results[0]):
            contexts.append({
                "rank": i + 1,
                "text": result.entity.text,
                "period": result.entity.period,
                "chunk_type": result.entity.chunk_type,
                "statement_type": result.entity.statement_type,
                "primary_value": result.entity.primary_value,
                "score": float(result.score)
            })
        return contexts
    
    def search_and_answer(self, question: str, top_k: int = 3):
        """RAG: Search + Generate answer"""
        contexts = self.search(question, top_k)
        context_str = "\n\n".join([
            f"Context {i+1} [{ctx['period']} - {ctx['chunk_type']}]:\n{ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""Based on the provided financial data about ICICI Bank, answer the question accurately.

QUESTION: {question}

CONTEXT:
{context_str}

INSTRUCTIONS:
- Use exact numbers from the context (include decimals and units)
- If information is not available, say so clearly
- Be concise and factual
- Include the relevant period/quarter

ANSWER:"""
        
        time.sleep(1)
        try:
            response = self.model.generate_content(prompt)
            return {
                "answer": response.text.strip(),
                "contexts": contexts,
                "context_count": len(contexts)
            }
        except Exception as e:
            return {
                "error": str(e),
                "contexts": contexts,
                "context_count": len(contexts)
            }
    
    def health_check(self):
        """Health check"""
        try:
            self.collection.num_entities
            test_response = self.model.generate_content("test")
            return {
                "status": "healthy",
                "milvus": "connected",
                "gemini": "available",
                "collection": COLLECTION_NAME,
                "total_chunks": self.collection.num_entities
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

# Initialize
rag = VectorRAG(GEMINI_API_KEY, COLLECTION_NAME)
mcp = FastMCP("VectorRAG")

@mcp.tool()
def health_check():
    """Check Vector RAG system health"""
    return rag.health_check()

@mcp.tool()
def search_vectors(query: str, top_k: int = 3):
    """Semantic search in vector store"""
    try:
        contexts = rag.search(query, top_k)
        return {
            "status": "success",
            "query": query,
            "results": contexts,
            "result_count": len(contexts)
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "query": query}

@mcp.tool()
def answer_question(question: str, top_k: int = 3):
    """Answer question using RAG"""
    try:
        result = rag.search_and_answer(question, top_k)
        return {"status": "success", "question": question, **result}
    except Exception as e:
        return {"status": "error", "message": str(e), "question": question}

@mcp.tool()
def get_collection_stats():
    """Get Milvus collection statistics"""
    try:
        return {
            "status": "success",
            "collection_name": COLLECTION_NAME,
            "total_chunks": rag.collection.num_entities,
            "milvus_host": MILVUS_HOST,
            "milvus_port": MILVUS_PORT
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    logger.info("Starting Vector RAG MCP Server")
    logger.info(f"Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    logger.info(f"Collection: {COLLECTION_NAME}")
    logger.info(f"Port: 9006")
    
    os.environ["PORT"] = "9006"
    mcp.run(transport="streamable-http")