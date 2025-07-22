# main_graph.py - Graph Construction Service Entry Point
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from graph.api import router as graph_router
from shared.config import get_settings, validate_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import argparse

# Add this function before the lifespan function:
def parse_args():
    parser = argparse.ArgumentParser(description='ICICI Graph Construction Service')
    parser.add_argument('--model', type=str, 
                       choices=['gemini-2.0-flash', 'gpt-3.5-turbo', 'llama3.1:8b', 'groq-llama'],
                       help='Model to use for entity extraction')
    return parser.parse_args()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 Graph Construction Service starting...")
    
    # Validate configuration
    if validate_config():
        logger.info("✅ Configuration validated")
    else:
        logger.warning("⚠️ Configuration issues detected")
    
    # Test Neo4j connection
    try:
        from graph.neo4j_service import Neo4jService
        settings = get_settings()
        
        neo4j = Neo4jService(
            settings.neo4j_uri,
            settings.neo4j_user,
            settings.neo4j_password
        )
        
        if neo4j.health_check():
            logger.info("✅ Neo4j connection established")
            
            # Get basic stats
            stats = neo4j.get_stats()
            if stats:
                total_nodes = sum(v for k, v in stats.items() if k.endswith('_count') and not k.startswith('HAS'))
                logger.info(f"📊 Graph contains {total_nodes} nodes")
            
        else:
            logger.error("❌ Neo4j connection failed")
            
        neo4j.close()
        
    except Exception as e:
        logger.error(f"❌ Neo4j setup failed: {e}")
    
    # Test entity extractor
    try:
        from entity.extraction import EntityExtractor
        # Test entity extractor initialization - FIXED
        settings = get_settings()
        
        # Use the same logic as entity service
        model_name = settings.default_model
        if "llama" in model_name:
            api_key = None  # Force local for llama
        else:
            api_key = settings.get_api_key_for_model(model_name)
        
        if api_key or "llama" in model_name:  # Allow llama without API key
            from entity.extraction import EntityExtractor
            extractor = EntityExtractor(model_name, api_key)
            logger.info(f"✅ Entity extractor ready with model: {extractor.current_model}")
        else:
            logger.warning(f"⚠️ No API key found for model: {model_name}")
    except Exception as e:
        logger.error(f"❌ Entity extractor test failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Graph Construction Service shutting down...")

# Create FastAPI app
settings = get_settings()

app = FastAPI(
    title="ICICI Graph Construction Service",
    description="Financial knowledge graph construction from ICICI Bank quarterly reports",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(graph_router, prefix="/api/v1", tags=["graph"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service info"""
    settings = get_settings()
    
    return {
        "service": "icici-graph-construction",
        "version": "1.0.0",
        "status": "running",
        "default_model": settings.default_model,
        "docs": "/docs",
        "health": "/api/v1/health",
        "neo4j_uri": settings.neo4j_uri,
        "endpoints": {
            "build_graph": "POST /api/v1/build",
            "query_graph": "POST /api/v1/query",
            "graph_stats": "GET /api/v1/stats",
            "clear_data": "DELETE /api/v1/clear/{dataset_id}",
            "graph_schema": "GET /api/v1/schema"
        },
        "example_usage": {
            "build": "POST /api/v1/build with chunks array",
            "query": "POST /api/v1/query with question string"
        }
    }

# Health check at root level
@app.get("/ping") 
async def ping():
    """Simple ping endpoint"""
    return {"status": "pong", "service": "graph-construction"}

# Also add at API level
@app.get("/api/v1/ping")
async def api_ping():
    """API level ping endpoint"""
    return {"status": "pong", "service": "graph-construction", "api_version": "v1"}

# Quick stats endpoint
@app.get("/quick-stats")
async def quick_stats():
    """Quick graph statistics without full health check"""
    try:
        from shared.config import get_graph_builder
        builder = get_graph_builder()
        
        if builder.is_healthy():
            stats = builder.get_stats()
            return {
                "neo4j_connected": True,
                "total_nodes": sum(v for k, v in stats.items() if k.endswith('_count') and not k.startswith('HAS')),
                "quarters": stats.get('Quarter_count', 0),
                "metrics": stats.get('Metric_count', 0),
                "current_model": builder.current_model
            }
        else:
            return {"neo4j_connected": False, "error": "Neo4j not accessible"}
            
    except Exception as e:
        return {"error": str(e), "neo4j_connected": False}

if __name__ == "__main__":
    args = parse_args()
    settings = get_settings()
    
    # Switch model if specified
    if args.model:
        from shared.config import switch_default_model
        switch_default_model(args.model)
        logger.info(f"🔄 Using model: {args.model}")
    
    # Use the port from settings (from .env or default)
    port = settings.graph_service_port
    
    logger.info(f"🚀 Starting Graph Construction Service on port {port}")
    logger.info(f"📖 API Documentation: http://localhost:{port}/docs")
    logger.info(f"🔗 Neo4j URI: {settings.neo4j_uri}")
    
    uvicorn.run(
        "main_graph:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level=settings.log_level.lower()
    )