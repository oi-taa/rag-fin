# graph/api.py - Simple and focused
from fastapi import APIRouter, HTTPException, Depends
from shared.models import GraphBuildRequest, GraphQueryRequest
from graph.graph_builder import GraphBuilder
from shared.config import get_graph_builder

router = APIRouter()

@router.get("/health")
async def health(builder: GraphBuilder = Depends(get_graph_builder)):
    """Simple health check"""
    return {
        "status": "healthy" if builder.is_healthy() else "unhealthy",
        "neo4j": builder.neo4j_connected(),
        "model": builder.current_model
    }

@router.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"status": "pong", "service": "graph-construction"}

@router.post("/build")
async def build_graph(request: GraphBuildRequest, 
                     builder: GraphBuilder = Depends(get_graph_builder)):
    """Build graph from chunks"""
    try:
        result = await builder.build(request.chunks, request.dataset_id, request.clear_existing)
        
        # Handle both dict and int returns for compatibility
        if isinstance(result, dict):
            return result
        else:
            # If it returns an int (old behavior), convert it
            return {
                "success": True,
                "message": f"Processed {result} chunks",
                "chunks_processed": result,
                "entities_created": 0,
                "dataset_id": request.dataset_id
            }
    except Exception as e:
        import traceback
        print(f"Build error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(500, f"Build failed: {e}")

@router.post("/query") 
async def query_graph(request: GraphQueryRequest,
                     builder: GraphBuilder = Depends(get_graph_builder)):
    """Query graph with natural language"""
    try:
        results = await builder.query(request.question, request.limit)
        return {"success": True, "results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")

@router.get("/stats")
async def get_stats(builder: GraphBuilder = Depends(get_graph_builder)):
    """Get graph statistics"""
    return builder.get_stats()

@router.delete("/clear/{dataset_id}")
async def clear_data(dataset_id: str, builder: GraphBuilder = Depends(get_graph_builder)):
    """Clear dataset"""
    builder.clear(dataset_id)
    return {"message": f"Cleared {dataset_id}"}