
import asyncio
import json
import re
import time
import logging
from shared.models import FinancialChunk, ExtractedEntities
from shared.model_providers import GeminiProvider, LlamaProvider, GPTProvider

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, delay: float = 4.0):
        self.delay = delay
        self.last_call = 0
    
    async def wait(self):
        elapsed = time.time() - self.last_call
        if elapsed < self.delay:
            await asyncio.sleep(self.delay - elapsed)
        self.last_call = time.time()

class EntityExtractor:
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = None, **kwargs):
        self.current_model, self.api_key = model_name, api_key
        if "gemini" in model_name: self.client = GeminiProvider(model_name, api_key)
        elif "llama" in model_name: self.client = LlamaProvider(model_name, api_key, **kwargs)
        elif "gpt" in model_name: self.client = GPTProvider(model_name, api_key)
        else: self.client = GeminiProvider("gemini-2.0-flash", api_key)
    
    

    async def extract(self, chunk: FinancialChunk) -> ExtractedEntities:
        try:
            prompt = self._build_prompt(chunk.text)
            response = await self.client.generate_content(prompt)
            
            if not response or not response.strip():
                return ExtractedEntities()
            
            # Clean response text
            response_text = response.strip()
            response_text = re.sub(r'```json\n?|```\n?', '', response_text)
            
            # Extract JSON from response (handles Llama's explanatory text)
            json_text = self._extract_json_from_response(response_text)
            if not json_text:
                return ExtractedEntities()
            
            # Clean floating point precision issues
            json_text = re.sub(r'(\d+)\.0{20,}', r'\1.0', json_text)
            json_text = re.sub(r'(\d+\.\d{1,2})\d{20,}', r'\1', json_text)
            
            try:
                parsed = json.loads(json_text)
                cleaned_data = self._clean_parsed_data(parsed)
                return ExtractedEntities(**cleaned_data)  # Fixed: use cleaned_data
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                return ExtractedEntities()
                
        except Exception as e:
            logger.error(f"Extraction failed for {chunk.id}: {e}")
            return ExtractedEntities()

    def _extract_json_from_response(self, response_text: str) -> str:
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        
        if json_start == -1 or json_end == -1 or json_start >= json_end:
            return ""
        
        return response_text[json_start:json_end + 1]

    def _clean_parsed_data(self, data: dict) -> dict:
        def filter_valid_items(items_list, required_field):
            if not items_list:
                return []
            return [item for item in items_list 
                    if item.get(required_field) is not None]
        
        return {
            'quarter': data.get('quarter'),
            'financial_metrics': filter_valid_items(data.get('financial_metrics', []), 'value'),
            'business_segments': filter_valid_items(data.get('business_segments', []), 'revenue'), 
            'financial_ratios': filter_valid_items(data.get('financial_ratios', []), 'value'),
            'balance_sheet_items': filter_valid_items(data.get('balance_sheet_items', []), 'value')
        }
    
    def switch_model(self, model_name: str, api_key: str = None, **kwargs):
        self.current_model, self.api_key = model_name, api_key or self.api_key
        if "gemini" in model_name: self.client = GeminiProvider(model_name, self.api_key)
        elif "llama" in model_name: self.client = LlamaProvider(model_name, api_key, **kwargs)
        elif "gpt" in model_name: self.client = GPTProvider(model_name, self.api_key)
    
    def _build_prompt(self, text: str) -> str:
        """Build comprehensive extraction prompt for ICICI data"""
        return f"""
        Extract ALL financial data from this ICICI Bank quarterly report text:
        
        {text}
        
        Return JSON with this EXACT schema:
        {{
            "quarter": "Q1_FY2024",
            "financial_metrics": [
                {{"name": "NET PROFIT", "value": 10636.0, "growth_yoy": 44.0, "unit": "crore"}},
                {{"name": "Operating Profit", "value": 15660.0, "unit": "crore"}},
                {{"name": "Total Income", "value": 52084.0, "growth_yoy": 32.8, "unit": "crore"}},
                {{"name": "Interest Income", "value": 37106.0, "unit": "crore"}},
                {{"name": "Other Income", "value": 14978.0, "unit": "crore"}},
                {{"name": "Total Expenses", "value": 36424.0, "unit": "crore"}},
                {{"name": "Interest Expenses", "value": 16368.0, "unit": "crore"}},
                {{"name": "Operating Expenses", "value": 20057.0, "unit": "crore"}},
                {{"name": "Provisions", "value": 1345.0, "unit": "crore"}}
            ],
            "business_segments": [
                {{"name": "RETAIL BANKING SEGMENT", "revenue": 31057.0, "margin": 13.5, "percentage_of_total": 35.5}},
                {{"name": "TREASURY SEGMENT", "revenue": 26306.0, "margin": 16.6, "percentage_of_total": 30.1}}
            ],
            "financial_ratios": [
                {{"name": "Basic EPS", "value": 15.22, "growth_yoy": 43.3, "unit": "per share"}},
                {{"name": "Diluted EPS", "value": 14.91, "unit": "per share"}},
                {{"name": "Net Margin", "value": 20.4, "unit": "percentage"}},
                {{"name": "Operating Margin", "value": 30.1, "unit": "percentage"}},
                {{"name": "Cost Ratio", "value": 69.9, "unit": "percentage"}}
            ],
            "balance_sheet_items": [
                {{"name": "Advances", "value": 1124875.0, "percentage_of_total": 55.1, "unit": "crore"}},
                {{"name": "Investments", "value": 692709.0, "percentage_of_total": 34.0, "unit": "crore"}},
                {{"name": "Customer Deposits", "value": 1269343.0, "unit": "crore"}},
                {{"name": "Total Assets", "value": 2039897.0, "unit": "crore"}},
                {{"name": "Total Equity", "value": 225150.0, "unit": "crore"}}
            ]
        }}

        EXTRACTION RULES:
        1. Extract EVERY financial number mentioned in the text
        2. Convert currency: ₹52,084 crore → 52084.0 
        3. Convert percentages: 20.4% → 20.4
        4. Extract growth rates: (+44.0% YoY) → 44.0
        5. Use exact names from text but standardize format
        6. Include ALL income items, expense items, ratios, segments, balance sheet items
        7. If percentage of total is mentioned, include it
        8. Return null for missing values, don't make up data
        
        INCOME STATEMENT ITEMS TO EXTRACT:
        - Total Income, Interest Income, Other Income
        - Total Expenses, Interest Expenses, Operating Expenses
        - Net Profit, Operating Profit, Provisions
        
        RATIOS TO EXTRACT:
        - EPS (Basic, Diluted), Margins (Net, Operating)
        - Cost Ratio, any other percentages mentioned
        
        BALANCE SHEET TO EXTRACT:
        - Advances, Investments, Deposits, Assets, Equity
        - Cash balances, Borrowings, Reserves
        
        SEGMENT DATA TO EXTRACT:
        - Revenue, Margin, Percentage of total for each segment
        
        Be comprehensive and extract EVERYTHING mentioned!
        Return only valid JSON, NO EXPLANATIONS ONLY VALID JSON!
        """

async def quick_extract(chunk_text: str, model: str = "gemini-2.0-flash", api_key: str = None) -> ExtractedEntities:
    """Quick extraction for testing"""
    chunk = FinancialChunk(id="test_chunk", period="Q1_FY2024", type="test", size=len(chunk_text), text=chunk_text)
    extractor = EntityExtractor(model, api_key)
    return await extractor.extract(chunk)