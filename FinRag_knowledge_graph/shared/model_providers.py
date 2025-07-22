
import asyncio
import time
import aiohttp
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    def __init__(self, model_name: str, api_key: str = None, rate_limit: float = 1.0):
        self.model_name = model_name
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.last_call = 0
    
    async def _rate_limit_wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.rate_limit:
            await asyncio.sleep(self.rate_limit - elapsed)
        self.last_call = time.time()
    
    @abstractmethod
    async def generate_content(self, prompt: str) -> str:
        pass

class GeminiProvider(LLMProvider):
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = None, rate_limit: float = 4.0):
        super().__init__(model_name, api_key, rate_limit)
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(model_name)
    
    async def generate_content(self, prompt: str) -> str:
        await self._rate_limit_wait()
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self.client.generate_content(prompt))
        return response.text

class LlamaProvider(LLMProvider):
    def __init__(self, model_name: str = "llama3.1:8b", api_key: str = None, 
                 base_url: str = "http://localhost:11434", rate_limit: float = 0.5):
        super().__init__(model_name, api_key, rate_limit)
        self.base_url = base_url
        
        self.use_groq = api_key is not None and api_key.strip() != ""
        
        print(f"🔍 LlamaProvider fixed:")
        print(f"   api_key: '{api_key}'")
        print(f"   api_key is not None: {api_key is not None}")
        print(f"   api_key.strip() != '': {api_key.strip() != '' if api_key else False}")
        print(f"   use_groq: {self.use_groq}")
    
    async def generate_content(self, prompt: str) -> str:
        await self._rate_limit_wait()
        
        if self.use_groq:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": "llama-3.1-70b-versatile",
                "temperature": 0.1,
                "max_tokens": 8192
            }
            async with aiohttp.ClientSession() as session:
                async with session.post("https://api.groq.com/openai/v1/chat/completions", 
                                      headers=headers, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        raise Exception(f"Groq error: {response.status}")
        else:
            payload = {"model": self.model_name, "prompt": prompt, "stream": False}
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        raise Exception(f"Ollama error: {response.status}")

class GPTProvider(LLMProvider):
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: str = None, rate_limit: float = 1.0):
        super().__init__(model_name, api_key, rate_limit)
        import openai
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def generate_content(self, prompt: str) -> str:
        await self._rate_limit_wait()
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=8192
        )
        return response.choices[0].message.content

class ModelFactory:
    @staticmethod
    def create_provider(model_type: str, api_key: str = None, **kwargs) -> LLMProvider:
        providers = {"gemini": GeminiProvider, "llama": LlamaProvider, "gpt": GPTProvider}
        if model_type not in providers:
            raise ValueError(f"Unknown model: {model_type}")
        return providers[model_type](api_key=api_key, **kwargs)