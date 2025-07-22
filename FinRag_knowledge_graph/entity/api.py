# entity/api.py - Simple entity extraction API
from fastapi import APIRouter, HTTPException
from typing import List
from shared.models import FinancialChunk, ExtractedEntities
from entity.extraction import EntityExtractor
from shared.config import get_settings

router = APIRouter()

# Global extractor instance
extractor = None

def get_extractor() -> EntityExtractor:
    global extractor
    if extractor is None:
        settings = get_settings()
        model_name = settings.default_model
        api_key = settings.get_api_key_for_model(model_name)
        extractor = EntityExtractor(model_name, api_key)
    return extractor

@router.get("/health")
async def health_check():
    """Health check for entity service"""
    try:
        ext = get_extractor()
        return {
            "status": "healthy",
            "model": ext.current_model,
            "api_configured": ext.api_key is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e)
        }

@router.post("/extract", response_model=ExtractedEntities)
async def extract_entities(chunk: FinancialChunk):
    """Extract entities from a single chunk"""
    try:
        ext = get_extractor()
        entities = await ext.extract(chunk)
        return entities
    except Exception as e:
        raise HTTPException(500, f"Extraction failed: {e}")

@router.post("/extract/batch")
async def extract_batch(chunks: List[FinancialChunk]):
    """Extract entities from multiple chunks"""
    try:
        ext = get_extractor()
        results = []
        
        for chunk in chunks:
            try:
                entities = await ext.extract(chunk)
                results.append({
                    "chunk_id": chunk.id,
                    "success": True,
                    "entities": entities
                })
            except Exception as e:
                results.append({
                    "chunk_id": chunk.id,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "total": len(chunks),
            "successful": sum(1 for r in results if r["success"]),
            "results": results
        }
    except Exception as e:
        raise HTTPException(500, f"Batch extraction failed: {e}")

@router.post("/model/switch")
async def switch_model(model_name: str):
    """Switch to different model for testing"""
    try:
        global extractor
        settings = get_settings()
        extractor = EntityExtractor(model_name, settings.gemini_api_key)
        
        return {
            "message": f"Switched to {model_name}",
            "current_model": model_name
        }
    except Exception as e:
        raise HTTPException(400, f"Model switch failed: {e}")

@router.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": [
            "gemini-2.0",
            "gemini-1.5", 
            "claude-3",
            "local-llm"
        ],
        "current_model": get_extractor().current_model
    }