import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections
import time

class SimpleRAG:
    def __init__(self, gemini_api_key: str):
        """Initialize RAG system with Gemini and Milvus vector store"""
        # Configure Gemini API
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Initialize sentence transformer for embeddings
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Connect to Milvus vector database
        connections.connect("default", host="localhost", port="19530")
        self.collection = Collection("fin_chunks")
        self.collection.load()
    
    def search_and_answer(self, question: str, top_k: int = 3):
        """Search vector store and generate answer using retrieved contexts"""
        print(f"Question: {question}")
        print("-" * 50)
        
        # Encode question and search vector store
        query_embedding = self.similarity_model.encode([question])
        results = self.collection.search(
            query_embedding,
            "embedding",
            {"metric_type": "COSINE"},
            top_k,
            output_fields=["text", "period", "chunk_type"]
        )
        
        # Extract and display retrieved contexts
        contexts = []
        print("📄 Retrieved Contexts:")
        for i, result in enumerate(results[0]):
            context = result.entity.text
            period = result.entity.period
            chunk_type = result.entity.chunk_type
            score = result.score
            
            contexts.append(context)
            print(f"\n{i+1}. [{period} - {chunk_type}] (Score: {score:.3f})")
            print(f"   {context[:200]}...")
        
        # Create prompt with retrieved contexts
        context_str = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""Based on the provided financial data about ICICI Bank, answer the question accurately.

QUESTION: {question}

CONTEXT:
{context_str}

INSTRUCTIONS:
- Use exact numbers from the context (include decimals and units)
- If information is not available, say so clearly
- Be concise and factual
- Include the relevant period/quarter

ANSWER:"""
        
        # Generate answer using Gemini
        print(f"\n🤖 Generating answer...")
        time.sleep(1)  # Rate limiting delay
        
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            print(f"\n✅ ANSWER:")
            print(f"{answer}")
            return answer
            
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            print(f"\n❌ {error_msg}")
            return error_msg

def main():
    """Main function to run RAG system tests and interactive mode"""
    # Initialize RAG system
    rag = SimpleRAG("AIzaSyAD9HkPZPEfZ7hdP6MpqeqoSx3WSE7dXbU")
    
    # Define test questions
    test_questions = [
        "What was ICICI Bank's net profit in Q1 FY2024?",
        "What was the operating margin for Q2 FY2024?",
        "How did retail banking perform in Q3 FY2024?",
        "What was the EPS for Q4 FY2024?",
        "What were the total assets in Q3 FY2024?"
    ]
    
    # Run automated tests
    print("🚀 Simple RAG System Test")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 Test {i}/{len(test_questions)}")
        rag.search_and_answer(question)
        print("\n" + "=" * 60)
        
        # Add delay between questions to avoid rate limiting
        if i < len(test_questions):
            time.sleep(2)
    
    print(f"\n✅ All tests completed!")
    
    # Start interactive mode
    print(f"\n💬 Interactive Mode (type 'quit' to exit):")
    while True:
        user_question = input("\nEnter question: ").strip()
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if user_question:
            rag.search_and_answer(user_question)
            print()

if __name__ == "__main__":
    main()