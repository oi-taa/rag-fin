 
"""
Constants and configuration data for the Graph RAG system
"""

FINANCIAL_ENTITY_TYPES = {
    "financial_metrics": [
        "NET PROFIT", "Operating Profit", "Total Income", "Interest Income", 
        "Other Income", "Total Expenses", "Interest Expenses", "Operating Expenses", "Provisions"
    ],
    "business_segments": [
        "RETAIL BANKING SEGMENT", "WHOLESALE BANKING SEGMENT", "TREASURY SEGMENT",
        "LIFE INSURANCE SEGMENT", "OTHERS SEGMENT"
    ],
    "financial_ratios": [
        "Basic EPS", "Diluted EPS", "Net Margin", "Operating Margin", "Cost Ratio"
    ],
    "balance_sheet_items": [
        "Advances", "Investments", "Customer Deposits", "Total Assets", "Total Equity",
        "Cash & RBI Balances", "Borrowings", "Share Capital", "Reserves & Surplus"
    ]
}

SUPPORTED_QUARTERS = ["Q1_FY2024", "Q2_FY2024", "Q3_FY2024", "Q4_FY2024"]

CHUNK_TYPES = [
    "profitability_analysis", "balance_sheet_analysis", 
    "financial_ratios", "segment_analysis"
]

SUPPORTED_MODELS = {
    "gemini-2.0-flash": {"rate_limit": 4.0, "max_tokens": 8192},
    "gemini-1.5-pro": {"rate_limit": 2.0, "max_tokens": 8192},
    "gpt-3.5-turbo": {"rate_limit": 1.0, "max_tokens": 8192},
    "llama3.1:8b": {"rate_limit": 0.5, "max_tokens": 4096},
    "groq-llama": {"rate_limit": 0.5, "max_tokens": 8192}
}

# Helper Functions for Data Validation
def validate_quarter(quarter: str) -> bool:
    """Validate quarter format"""
    return quarter in SUPPORTED_QUARTERS

def validate_chunk_type(chunk_type: str) -> bool:
    """Validate chunk type"""
    return chunk_type in CHUNK_TYPES