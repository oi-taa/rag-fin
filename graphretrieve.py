# test_graph_only.py
from graph_cons import FinancialHybridRAG
import json
import google.generativeai as genai
import re



# Test class
class GraphOnlyTester:
    def __init__(self):
        self.rag = FinancialHybridRAG(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            milvus_collection_name="fin_chunks"
        )
    
    def test_graph_search(self, question):
        print(f"Testing graph search for: '{question}'")
        
        
        results = self.rag.graph_search(question)
        print(f"Graph results: {len(results)} found")
        
        # Show first few results
        for i, result in enumerate(results[:3]):
            print(f"  Result {i+1}: {result}")
        
        return results
    
    def answer_question_with_graph(self, question):
        """Complete pipeline: Graph search + LLM generation"""
        
        # 1. Get structured data from graph
        graph_results = self.graph_search(question)
        
        # 2. Format context for LLM
        context = self._format_graph_results(graph_results)
        
        # 3. Generate natural language answer
        prompt = f"""
        Question: {question}
        
        Financial Data from Knowledge Graph:
        {context}
        
        Provide a clear, accurate answer based on this data. Include specific numbers and growth rates.
        """
        
        response = self.model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text

    def _format_graph_results(self, results):
        """Format graph results for LLM context"""
        formatted = []
        
        for result in results:
            if 'metric_name' in result:
                formatted.append(f"- {result['quarter']}: {result['metric_name']} = ₹{result['value']} {result['unit']} (Growth: {result['growth']}% YoY)")
            elif 'segment_name' in result:
                formatted.append(f"- {result['quarter']}: {result['segment_name']} Revenue = ₹{result['revenue']} crore (Margin: {result['margin']}%)")
        
        return "\n".join(formatted)

# Run the test
if __name__ == "__main__":
    tester = GraphOnlyTester()
    
    questions = [
        "How have margins changed over time?", 
        "What's the net profit trend?",
        "Analyze corporate banking margins", 
        "How much revenue did life insurance generate in Q3?"
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        results = tester.test_graph_search(q)
        print(f"Total results: {len(results)}")