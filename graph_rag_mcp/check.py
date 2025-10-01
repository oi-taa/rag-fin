# test_mcp_endpoints.py
import requests

base_url = "http://localhost:9007"

# Try different endpoint patterns
patterns = [
    f"{base_url}/",
    f"{base_url}/tools/extract_financial_entities",
    f"{base_url}/call/extract_financial_entities",
    f"{base_url}/mcp/tools/extract_financial_entities",
    f"{base_url}/sse",
]

for pattern in patterns:
    try:
        print(f"\nTrying: {pattern}")
        response = requests.get(pattern, timeout=2)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.text[:200]}")
    except Exception as e:
        print(f"  Error: {e}")

# Also try POST
print("\n--- Trying POST ---")
try:
    response = requests.post(
        f"{base_url}/tools/extract_financial_entities",
        json={"text": "test", "period": "Q1_FY2024"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")