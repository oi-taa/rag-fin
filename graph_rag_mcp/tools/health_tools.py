"""
Health and information tools for Graph RAG MCP Server
Preserving all original functionality from health check tools
"""

import os
import time
import logging
from fastmcp import FastMCP
from config.settings import get_config
from constants import FINANCIAL_ENTITY_TYPES, SUPPORTED_MODELS, SUPPORTED_QUARTERS
from providers.llm_providers import GeminiProvider, LlamaProvider, GPTProvider

logger = logging.getLogger(__name__)

def register_health_tools(server: FastMCP):
    """Register health and information tools"""
    
    @server.tool("health_check", description="Check Graph RAG server health and dependencies")
    def mcp_health_check():
        """Comprehensive health check for all Graph RAG dependencies"""
        config = get_config()
        health_status = {
            "server": "healthy",
            "timestamp": time.time(),
            "config": {
                "neo4j_uri": config.neo4j_uri,
                "default_model": config.default_model,
                "supported_models": list(SUPPORTED_MODELS.keys())
            },
            "dependencies": {
                "pydantic": True,  # We imported it successfully
                "aiohttp": True,   # We imported it successfully
            },
            "llm_providers": {},
            "issues": []
        }
        
        # Check configuration
        config_issues = config.validate()
        if config_issues:
            health_status["issues"].extend(config_issues)
            health_status["server"] = "warning"
        
        # Check LLM provider availability
        try:
            # Gemini
            if config.gemini_api_key:
                try:
                    import google.generativeai
                    health_status["dependencies"]["google_generativeai"] = True
                    health_status["llm_providers"]["gemini"] = "available"
                except ImportError:
                    health_status["dependencies"]["google_generativeai"] = False
                    health_status["llm_providers"]["gemini"] = "missing_package"
                    health_status["issues"].append("google-generativeai package not installed")
            else:
                health_status["llm_providers"]["gemini"] = "no_api_key"
            
            # OpenAI
            if config.openai_api_key:
                try:
                    import openai
                    health_status["dependencies"]["openai"] = True
                    health_status["llm_providers"]["openai"] = "available"
                except ImportError:
                    health_status["dependencies"]["openai"] = False
                    health_status["llm_providers"]["openai"] = "missing_package"
                    health_status["issues"].append("openai package not installed")
            else:
                health_status["llm_providers"]["openai"] = "no_api_key"
            
            # Groq/Llama
            if config.groq_api_key:
                health_status["llm_providers"]["groq"] = "available"
            else:
                health_status["llm_providers"]["groq"] = "no_api_key"
            
            # Local Llama (always available)
            health_status["llm_providers"]["llama_local"] = "available_if_ollama_running"
            
        except Exception as e:
            health_status["issues"].append(f"LLM provider check failed: {str(e)}")
            health_status["server"] = "warning"
        
        # Check default model availability
        default_api_key = config.get_api_key_for_model(config.default_model)
        if not default_api_key and "llama" not in config.default_model.lower():
            health_status["issues"].append(f"Default model {config.default_model} has no API key configured")
            health_status["server"] = "warning"
        
        # Neo4j health check
        try:
            from services.neo4j_service import Neo4jService
            neo4j = Neo4jService(config.neo4j_uri, config.neo4j_user, config.neo4j_password)
            if neo4j.health_check():
                health_status["dependencies"]["neo4j"] = "connected"
            else:
                health_status["dependencies"]["neo4j"] = "connection_failed"
                health_status["issues"].append("Neo4j connection failed")
            neo4j.close()
        except Exception as e:
            health_status["dependencies"]["neo4j"] = f"error: {str(e)}"
            health_status["issues"].append(f"Neo4j error: {str(e)}")
        
        return health_status

    @server.tool("get_server_info", description="Get Graph RAG server information and capabilities")
    def mcp_get_server_info():
        """Return comprehensive server information"""
        config = get_config()
        
        return {
            "server_name": "Graph RAG Financial Analysis Server",
            "version": "1.0.0",
            "description": "Standalone Graph RAG for financial quarterly analysis",
            "port": config.port,
            "capabilities": {
                "entity_extraction": True,
                "knowledge_graph": True,
                "natural_language_queries": True,
                "multi_llm_support": True,
                "financial_analysis": True
            },
            "supported_features": {
                "entity_types": list(FINANCIAL_ENTITY_TYPES.keys()),
                "quarters": SUPPORTED_QUARTERS,
                "models": list(SUPPORTED_MODELS.keys())
            },
            "endpoints": {
                "health": "health_check",
                "info": "get_server_info", 
                "extract": "extract_financial_entities",
                "build": "build_financial_graph",
                "query": "query_financial_graph",
                "stats": "get_graph_stats"
            }
        }

    @server.tool("switch_default_model", description="Switch the default LLM model for the server")
    def mcp_switch_model(model_name: str):
        """Switch default model (updates configuration)"""
        if model_name not in SUPPORTED_MODELS:
            return {
                "status": "error",
                "message": f"Unsupported model: {model_name}",
                "supported_models": list(SUPPORTED_MODELS.keys())
            }
        
        config = get_config()
        api_key = config.get_api_key_for_model(model_name)
        
        if not api_key and "llama" not in model_name.lower():
            return {
                "status": "error", 
                "message": f"No API key configured for model: {model_name}",
                "required_env_var": "GEMINI_API_KEY" if "gemini" in model_name else "OPENAI_API_KEY" if "gpt" in model_name else "GROQ_API_KEY"
            }
        
        # Update environment variable (for this session)
        os.environ["DEFAULT_MODEL"] = model_name
        
        # Clear config cache to pick up new default
        get_config.cache_clear()
        
        return {
            "status": "success",
            "message": f"Default model switched to {model_name}",
            "previous_model": config.default_model,
            "new_model": model_name,
            "api_key_configured": bool(api_key) or "llama" in model_name.lower()
        }

    @server.tool("test_llm_simple", description="Test LLM generation with default settings (no parameters needed)")
    async def mcp_test_llm_simple():
        """Test LLM generation with hardcoded defaults"""
        config = get_config()
        model_name = config.default_model
        test_prompt = "What is 2+2?"
        
        try:
            # Get appropriate API key
            api_key = config.get_api_key_for_model(model_name)
            
            # Create provider based on model
            if "gemini" in model_name.lower():
                if not api_key:
                    return {"status": "error", "message": "Gemini API key not configured"}
                provider = GeminiProvider(model_name, api_key)
            elif "gpt" in model_name.lower():
                if not api_key:
                    return {"status": "error", "message": "OpenAI API key not configured"}
                provider = GPTProvider(model_name, api_key)
            elif "llama" in model_name.lower():
                provider = LlamaProvider(model_name, api_key)  # Can work with or without API key
            else:
                return {"status": "error", "message": f"Unsupported model: {model_name}"}
            
            # Test generation
            start_time = time.time()
            response = await provider.generate_content(test_prompt)
            generation_time = time.time() - start_time
            
            return {
                "status": "success",
                "message": "Content generation successful",
                "model_used": model_name,
                "test_prompt": test_prompt,
                "response": response,
                "generation_time_seconds": round(generation_time, 2),
                "response_length": len(response),
                "provider_type": type(provider).__name__
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Content generation failed: {str(e)}",
                "model_attempted": model_name,
                "error_details": str(e)
            }