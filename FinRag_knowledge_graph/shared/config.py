# shared/config.py - Ultra compact with all functionality
import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field

MODEL_CONFIGS = {
    "gemini-2.0-flash": {"rate_limit": 4.0, "max_tokens": 8192},
    "gemini-1.5-pro": {"rate_limit": 2.0, "max_tokens": 8192},
    "gpt-3.5-turbo": {"rate_limit": 1.0, "max_tokens": 8192},
    "llama3.1:8b": {"rate_limit": 0.5, "max_tokens": 4096},
    "groq-llama": {"rate_limit": 0.5, "max_tokens": 8192},
    "claude-3-sonnet": {"rate_limit": 1.0, "max_tokens": 4096}
}

class Settings(BaseSettings):
    # Neo4j
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="password", env="NEO4J_PASSWORD")
    
    # API Keys
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    
    # Services
    graph_service_port: int = Field(default=8002, env="GRAPH_SERVICE_PORT")
    entity_service_port: int = Field(default=8001, env="ENTITY_SERVICE_PORT")
    default_model: str = Field(default="gemini-2.0-flash", env="DEFAULT_MODEL")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    model_config = {"env_file": ".env", "case_sensitive": False, "protected_namespaces": ()}
    
    def get_api_key_for_model(self, model_name: str) -> str:
        if "gemini" in model_name: return self.gemini_api_key
        elif "gpt" in model_name: return self.openai_api_key
        elif "groq" in model_name: return self.groq_api_key
        elif model_name == "llama3.1:8b": return ""
        else: return ""

@lru_cache()
def get_settings() -> Settings:
    return Settings()

def get_graph_builder():
    from graph.graph_builder import GraphBuilder
    settings = get_settings()
    api_key = settings.get_api_key_for_model(settings.default_model)
    return GraphBuilder(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, 
                       api_key, settings.default_model)

def get_entity_extractor(model_name: str = None):
    from entity.extraction import EntityExtractor
    settings = get_settings()
    model = model_name or settings.default_model
    return EntityExtractor(model, settings.get_api_key_for_model(model))

def switch_default_model(model_name: str):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    os.environ["DEFAULT_MODEL"] = model_name
    get_settings.cache_clear()
    print(f"✅ Switched default model to: {model_name}")

def validate_config():
    settings = get_settings()
    issues = []
    
    if not settings.gemini_api_key: issues.append("❌ GEMINI_API_KEY not set")
    if not settings.openai_api_key: issues.append("⚠️ OPENAI_API_KEY not set")
    if not settings.groq_api_key: issues.append("⚠️ GROQ_API_KEY not set")
    if settings.neo4j_password == "password": issues.append("⚠️ Using default Neo4j password")
    if settings.default_model not in MODEL_CONFIGS: issues.append(f"❌ Invalid model: {settings.default_model}")
    
    api_key = settings.get_api_key_for_model(settings.default_model)
    if not api_key and settings.default_model != "llama3.1:8b":
        issues.append(f"❌ No API key for: {settings.default_model}")
    
    if issues:
        print("Configuration Issues:")
        for issue in issues: print(f"  {issue}")
        return False
    else:
        print("✅ Configuration OK!")
        return True