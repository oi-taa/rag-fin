# shared/models.py - Common data models used across services
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

# ===============================
# CHUNK MODELS
# ===============================

class ChunkType(str, Enum):
    BALANCE_SHEET = "balance_sheet_analysis"
    FINANCIAL_RATIOS = "financial_ratios" 
    PROFITABILITY = "profitability_analysis"
    SEGMENT_ANALYSIS = "segment_analysis"

class FinancialChunk(BaseModel):
    """Financial chunk from ICICI quarterly reports"""
    id: str
    period: str = Field(..., pattern=r"Q[1-4]_FY\d{4}")  # e.g., Q1_FY2024
    type: str  # Using str instead of enum for flexibility
    size: int
    text: str = Field(..., min_length=10)

# ===============================
# ENTITY MODELS
# ===============================

class FinancialMetric(BaseModel):
    """Financial metric (profit, revenue, etc.)"""
    name: str
    value: float
    growth_yoy: Optional[float] = None
    unit: str = "crore"

class BusinessSegment(BaseModel):
    """Business segment performance"""
    name: str
    revenue: float
    margin: float
    percentage_of_total: Optional[float] = None

class FinancialRatio(BaseModel):
    """Financial ratio (EPS, margins, etc.)"""
    name: str
    value: float
    growth_yoy: Optional[float] = None
    unit: str = "ratio"

class BalanceSheetItem(BaseModel):
    """Balance sheet item"""
    name: str
    value: float
    percentage_of_total: Optional[float] = None
    unit: str = "crore"

class ExtractedEntities(BaseModel):
    """All entities extracted from a chunk"""
    quarter: Optional[str] = None
    financial_metrics: List[FinancialMetric] = []
    business_segments: List[BusinessSegment] = []
    financial_ratios: List[FinancialRatio] = []
    balance_sheet_items: List[BalanceSheetItem] = []

# ===============================
# API REQUEST/RESPONSE MODELS
# ===============================

class GraphBuildRequest(BaseModel):
    """Request to build knowledge graph"""
    chunks: List[FinancialChunk]
    dataset_id: str = "icici_fy2024"
    clear_existing: bool = False

class GraphBuildResponse(BaseModel):
    """Response from graph building"""
    success: bool
    message: str
    chunks_processed: int
    entities_created: int = 0
    relationships_created: int = 0
    dataset_id: str

class GraphQueryRequest(BaseModel):
    """Request to query knowledge graph"""
    question: str = Field(..., min_length=5)
    limit: int = Field(10, ge=1, le=100)

class GraphQueryResponse(BaseModel):
    """Response from graph query"""
    success: bool
    question: str
    results: List[Dict[str, Any]]
    result_count: int
    execution_time_ms: float = 0.0

class HealthResponse(BaseModel):
    """Health check response"""
    status: str  # "healthy" or "unhealthy"
    neo4j_connected: bool = False
    entity_service_available: bool = False
    version: str = "1.0.0"

class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: Optional[float] = None

# ===============================
# EVALUATION MODELS
# ===============================

class ModelComparison(BaseModel):
    """Model comparison results"""
    chunk_id: str
    models_tested: List[str]
    results: Dict[str, ExtractedEntities]
    best_model: Optional[str] = None
    notes: Optional[str] = None

class EvaluationMetrics(BaseModel):
    """Evaluation metrics for model performance"""
    model_name: str
    total_chunks: int
    successful_extractions: int
    avg_processing_time: float
    entities_per_chunk: float
    accuracy_score: Optional[float] = None

# ===============================
# DATASET MODELS
# ===============================

class Dataset(BaseModel):
    """Dataset information"""
    id: str
    name: str
    description: Optional[str] = None
    total_chunks: int
    quarters: List[str]
    created_at: Optional[str] = None

class DatasetStats(BaseModel):
    """Dataset statistics"""
    dataset_id: str
    total_nodes: int
    total_relationships: int
    quarters_count: int
    metrics_count: int
    segments_count: int

