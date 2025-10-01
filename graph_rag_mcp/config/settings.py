"""
Configuration management for Graph RAG MCP Server
"""

import os
from typing import List
from dataclasses import dataclass
from functools import lru_cache

@dataclass
class GraphRAGConfig:
    """Configuration for Graph RAG MCP Server"""
    # Neo4j Configuration
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")

    # LLM API Keys
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "AIzaSyBhPk2fEF7tDNlaWZxv5oECp26-lBCiuLk")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    
    # Default Settings
    default_model: str = os.getenv("DEFAULT_MODEL", "gemini-2.0-flash") # swap with llama3.1:8b if using llama remove key
    
    # Base timeouts
    base_graph_build_timeout: int = int(os.getenv("GRAPH_BUILD_TIMEOUT", "300"))
    base_entity_extraction_timeout: int = int(os.getenv("ENTITY_EXTRACTION_TIMEOUT", "60"))
    base_graph_query_timeout: int = int(os.getenv("GRAPH_QUERY_TIMEOUT", "30"))
    
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Server Configuration
    port: int = int(os.getenv("GRAPH_RAG_PORT", "9007"))
    host: str = os.getenv("GRAPH_RAG_HOST", "0.0.0.0")
    
    @property
    def graph_build_timeout(self) -> int:
        """Get timeout based on current model"""
        if "llama" in self.default_model.lower():
            return self.base_graph_build_timeout * 2
        return self.base_graph_build_timeout
    
    @property 
    def entity_extraction_timeout(self) -> int:
        """Get extraction timeout based on current model"""
        if "llama" in self.default_model.lower():
            return self.base_entity_extraction_timeout * 2
        return self.base_entity_extraction_timeout
    
    @property
    def graph_query_timeout(self) -> int:
        """Get query timeout based on current model"""
        if "llama" in self.default_model.lower():
            return self.base_graph_query_timeout * 2
        return self.base_graph_query_timeout
    
    def get_api_key_for_model(self, model_name: str) -> str:
        """Get appropriate API key for model"""
        if "gemini" in model_name.lower():
            return self.gemini_api_key
        elif "gpt" in model_name.lower() or "openai" in model_name.lower():
            return self.openai_api_key
        elif "groq" in model_name.lower():
            return self.groq_api_key
        elif "llama" in model_name.lower():
            return ""  # Local model, no API key needed
        else:
            return ""

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check Neo4j configuration
        if not self.neo4j_uri:
            issues.append("NEO4J_URI not configured")
        if self.neo4j_password == "password":
            issues.append("Using default Neo4j password - consider changing for security")
        
        # Check API keys
        if not self.gemini_api_key and "gemini" in self.default_model:
            issues.append(f"GEMINI_API_KEY required for default model: {self.default_model}")
        if not self.openai_api_key and "gpt" in self.default_model:
            issues.append(f"OPENAI_API_KEY required for default model: {self.default_model}")
        
        return issues

# Global configuration instance
@lru_cache()
def get_config() -> GraphRAGConfig:
    return GraphRAGConfig()