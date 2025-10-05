"""
Vector RAG Adapter - Simple REST to MCP Bridge
Port: 9001 -> MCP Server: 9006

EXACT COPY of graphrag_adapter.py pattern
"""

import os
import json
import logging
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
MCP_URL = os.getenv("VECTOR_RAG_MCP_URL", "http://localhost:9006/mcp")
TIMEOUT = 300

# Pydantic Models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=5)
    top_k: int = Field(default=3, ge=1, le=20)

class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=5)
    top_k: int = Field(default=3, ge=1, le=10)

# MCP Client - SIMPLE SSE HANDLER (EXACT copy from graphrag_adapter)
class MCPClient:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=TIMEOUT)
        self.session_id = None
    
    async def init_session(self):
        """Initialize MCP session once"""
        if self.session_id:
            return
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }
        
        init_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "vectorrag-adapter", "version": "1.0.0"}
            }
        }
        
        response = await self.client.post(MCP_URL, json=init_msg, headers=headers)
        self.session_id = response.headers.get("mcp-session-id")
        
        # Send initialized notification
        await self.client.post(
            MCP_URL,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers={**headers, "mcp-session-id": self.session_id}
        )
        
        logger.info(f"Session initialized: {self.session_id}")
    
    async def call_tool(self, tool_name: str, args: dict) -> dict:
        """Call MCP tool and handle SSE response"""
        await self.init_session()
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": self.session_id
        }
        
        message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": args}
        }
        
        logger.info(f"Calling tool: {tool_name}")
        
        # Stream response (SSE format)
        async with self.client.stream("POST", MCP_URL, json=message, headers=headers) as response:
            if response.status_code != 200:
                raise HTTPException(503, f"MCP error: {response.status_code}")
            
            # Parse SSE stream
            result = None
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    try:
                        parsed = json.loads(data)
                        if "result" in parsed:
                            result = parsed["result"]
                            break
                        if "error" in parsed:
                            raise HTTPException(500, f"Tool error: {parsed['error']}")
                    except json.JSONDecodeError:
                        continue
            
            if result is None:
                raise HTTPException(500, "No result from MCP")
            
            return result

# Global client
mcp = MCPClient()

# FastAPI App
app = FastAPI(title="Vector RAG Adapter")

@app.get("/")
def root():
    return {
        "service": "vector-rag-adapter",
        "mcp_server": MCP_URL,
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "answer": "/answer",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health():
    try:
        result = await mcp.call_tool("health_check", {})
        return {"status": "healthy", "mcp": result}
    except:
        return {"status": "unhealthy", "mcp": "unavailable"}

@app.post("/search")
async def search(req: SearchRequest):
    """Semantic vector search"""
    result = await mcp.call_tool("search_vectors", {
        "query": req.query,
        "top_k": req.top_k
    })
    return result

@app.post("/answer")
async def answer(req: AnswerRequest):
    """Answer question using RAG"""
    result = await mcp.call_tool("answer_question", {
        "question": req.question,
        "top_k": req.top_k
    })
    return result

@app.get("/stats")
async def stats():
    """Get collection stats"""
    result = await mcp.call_tool("get_collection_stats", {})
    return result

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Vector RAG Adapter on port 9001")
    logger.info(f"🔗 MCP Server: {MCP_URL}")
    uvicorn.run(app, host="0.0.0.0", port=9001)