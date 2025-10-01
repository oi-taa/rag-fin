"""
GraphRAG Adapter - Simple REST to MCP Bridge
Port: 9002 -> MCP Server: 9007

SIMPLE VERSION - handles SSE streaming properly
"""

import os
import json
import logging
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
MCP_URL = "http://localhost:9007/mcp"
TIMEOUT = 300

# Pydantic Models
class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=10)
    period: Optional[str] = None
    model: Optional[str] = None

class BuildGraphRequest(BaseModel):
    chunks: List[Dict[str, Any]]
    dataset_id: str = "graphrag_data"
    clear_existing: bool = False

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5)
    limit: int = Field(default=10, ge=1, le=100)

# MCP Client - SIMPLE SSE HANDLER
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
                "clientInfo": {"name": "graphrag-adapter", "version": "1.0.0"}
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
app = FastAPI(title="GraphRAG Adapter")

@app.get("/")
def root():
    return {
        "service": "graphrag-adapter",
        "mcp_server": MCP_URL,
        "endpoints": {
            "health": "/health",
            "extract": "/extract",
            "build": "/build",
            "query": "/query",
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

@app.post("/extract")
async def extract(req: ExtractRequest):
    result = await mcp.call_tool("extract_financial_entities", {
        "text": req.text,
        "period": req.period,
        "model": req.model
    })
    return result

@app.post("/build")
async def build(req: BuildGraphRequest):
    result = await mcp.call_tool("build_financial_graph", {
        "chunks": req.chunks,
        "dataset_id": req.dataset_id,
        "clear_existing": req.clear_existing
    })
    return result

@app.post("/query")
async def query(req: QueryRequest):
    result = await mcp.call_tool("query_financial_graph", {
        "question": req.question,
        "limit": req.limit
    })
    return result

@app.get("/stats")
async def stats():
    result = await mcp.call_tool("get_graph_stats", {})
    return result

@app.delete("/clear")
async def clear(dataset_id: Optional[str] = None):
    result = await mcp.call_tool("clear_graph_data", {"dataset_id": dataset_id})
    return result

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 GraphRAG Adapter on port 9002")
    logger.info(f"🔗 MCP Server: {MCP_URL}")
    uvicorn.run(app, host="0.0.0.0", port=9002)