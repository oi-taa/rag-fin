"""
Vector RAG System Test & Inspection Script

Tests:
1. Milvus connection and data
2. MCP server tools
3. Adapter endpoints
"""

import requests
import json
from pymilvus import connections, Collection

print("=" * 60)
print("VECTOR RAG SYSTEM INSPECTION")
print("=" * 60)

# ========================================
# TEST 1: Milvus Direct Connection
# ========================================
print("\n📊 TEST 1: Milvus Collection Check")
print("-" * 60)

try:
    connections.connect("default", host="localhost", port="19530")
    collection = Collection("fin_chunks")
    collection.load()
    
    num_entities = collection.num_entities
    print(f"✅ Milvus Connected")
    print(f"   Collection: fin_chunks")
    print(f"   Total chunks: {num_entities}")
    
    # Get sample data
    results = collection.query(
        expr="",
        limit=3,
        output_fields=["id", "period", "chunk_type", "statement_type"]
    )
    
    print(f"\n   Sample chunks:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result['id']}")
        print(f"      Period: {result['period']}")
        print(f"      Type: {result['chunk_type']}")
    
except Exception as e:
    print(f"❌ Milvus Error: {e}")

# ========================================
# TEST 2: MCP Server (Port 9006)
# ========================================
print("\n\n🔧 TEST 2: MCP Server Tools (Port 9006)")
print("-" * 60)
print("Use MCP Inspector to test:")
print("   npx @modelcontextprotocol/inspector http://localhost:9006/mcp")
print("\nAvailable Tools:")
print("   1. health_check")
print("   2. search_vectors")
print("   3. answer_question")
print("   4. get_collection_stats")

# ========================================
# TEST 3: Adapter REST API (Port 9001)
# ========================================
print("\n\n🌐 TEST 3: Adapter REST Endpoints (Port 9001)")
print("-" * 60)

# Test Health
print("\n1. Testing /health endpoint...")
try:
    response = requests.get("http://localhost:9001/health", timeout=5)
    if response.status_code == 200:
        print(f"   ✅ Health check passed")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"   ❌ Status: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    print(f"   Make sure adapter is running on port 9001")

# Test Stats
print("\n2. Testing /stats endpoint...")
try:
    response = requests.get("http://localhost:9001/stats", timeout=5)
    if response.status_code == 200:
        print(f"   ✅ Stats retrieved")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"   ❌ Status: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test Search
print("\n3. Testing /search endpoint...")
try:
    search_data = {
        "query": "net profit Q1",
        "top_k": 3
    }
    response = requests.post(
        "http://localhost:9001/search",
        json=search_data,
        timeout=10
    )
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Search successful")
        print(f"   Query: {search_data['query']}")
        print(f"   Results found: {result.get('result_count', 0)}")
        if 'results' in result:
            for i, ctx in enumerate(result['results'][:2], 1):
                print(f"\n   Result {i}:")
                print(f"      Period: {ctx.get('period')}")
                print(f"      Type: {ctx.get('chunk_type')}")
                print(f"      Score: {ctx.get('score', 0):.3f}")
                print(f"      Text: {ctx.get('text', '')[:100]}...")
    else:
        print(f"   ❌ Status: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test Answer
print("\n4. Testing /answer endpoint...")
try:
    answer_data = {
        "question": "What was ICICI Bank's net profit in Q1 FY2024?",
        "top_k": 3
    }
    response = requests.post(
        "http://localhost:9001/answer",
        json=answer_data,
        timeout=30
    )
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Answer generated")
        print(f"   Question: {answer_data['question']}")
        print(f"\n   Answer:")
        print(f"   {result.get('answer', 'No answer')}")
        print(f"\n   Contexts used: {result.get('context_count', 0)}")
    else:
        print(f"   ❌ Status: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# ========================================
# SUMMARY
# ========================================
print("\n\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)
print("\nNext Steps:")
print("1. If Milvus failed: Run chunking_storing.py first")
print("2. If MCP failed: Start vector_rag_mcp/main.py")
print("3. If Adapter failed: Start adapters/vector_rag_adapter.py")
print("\nMCP Inspector Command:")
print("   npx @modelcontextprotocol/inspector http://localhost:9006/mcp")
print("\nManual Test Commands:")
print("   # Health Check")
print("   curl http://localhost:9001/health")
print("\n   # Search")
print('   curl -X POST http://localhost:9001/search \\')
print('     -H "Content-Type: application/json" \\')
print('     -d \'{"query": "net profit Q1", "top_k": 3}\'')
print("\n   # Answer")
print('   curl -X POST http://localhost:9001/answer \\')
print('     -H "Content-Type: application/json" \\')
print('     -d \'{"question": "What was ICICI net profit in Q1 FY2024?", "top_k": 3}\'')
print("=" * 60)