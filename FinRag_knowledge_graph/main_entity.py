# main_entity.py - Entity Extraction Service Entry Point
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from entity.api import router as entity_router
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
    logger.info("🚀 Entity Extraction Service starting...")
    
    # Validate configuration
    if validate_config():
        logger.info("✅ Configuration validated")
    else:
        logger.warning("⚠️ Configuration issues detected")
    
    # Test entity extractor initialization - UPDATED
    try:
        settings = get_settings()
        
        # Get the current default model (respects CLI arguments)
        model_name = settings.default_model
        api_key = settings.get_api_key_for_model(model_name)
        
        if api_key or "llama" in model_name:  # Llama doesn't need API key
            from entity.extraction import EntityExtractor
            extractor = EntityExtractor(model_name, api_key)
            logger.info(f"✅ Entity extractor ready with model: {extractor.current_model}")
        else:
            logger.warning(f"⚠️ No API key found for model: {model_name} - entity extraction disabled")
    except Exception as e:
        logger.error(f"❌ Entity extractor initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Entity Extraction Service shutting down...")

# Create FastAPI app
settings = get_settings()

app = FastAPI(
    title="ICICI Entity Extraction Service",
    description="Financial entity extraction using configurable AI models",
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
app.include_router(entity_router, prefix="/api/v1", tags=["entity"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service info"""
    settings = get_settings()
    
    return {
        "service": "icici-entity-extraction",
        "version": "1.0.0",
        "status": "running",
        "default_model": settings.default_model,
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "extract_single": "POST /api/v1/extract",
            "extract_batch": "POST /api/v1/extract/batch", 
            "switch_model": "POST /api/v1/model/switch",
            "list_models": "GET /api/v1/models"
        }
    }

# Health check at root level
@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"status": "pong", "service": "entity-extraction"}

if __name__ == "__main__":
    args = parse_args()
    settings = get_settings()
    
    # Switch model if specified
    if args.model:
        from shared.config import switch_default_model
        switch_default_model(args.model)
        logger.info(f"🔄 Using model: {args.model}")
    
    
    logger.info(f"🚀 Starting Entity Extraction Service on port {settings.entity_service_port}")
    logger.info(f"📖 API Documentation: http://localhost:{settings.entity_service_port}/docs")
    
    uvicorn.run(
        "main_entity:app",
        host="0.0.0.0",
        port=settings.entity_service_port,
        reload=True,
        log_level=settings.log_level.lower()
    )