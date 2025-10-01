# test_tool_format.py
import httpx
import asyncio

async def test():
    url = "http://localhost:9007/mcp"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Initialize
        init = {"jsonrpc": "2.0", "id": 0, "method": "initialize", 
                "params": {"protocolVersion": "2024-11-05", "capabilities": {}, 
                          "clientInfo": {"name": "test", "version": "1.0.0"}}}
        response = await client.post(url, json=init, headers={"Accept": "application/json, text/event-stream"})
        session_id = response.headers.get("mcp-session-id")
        
        # Send initialized notification
        initialized = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        await client.post(url, json=initialized, headers={"mcp-session-id": session_id})
        
        # Wait a bit
        await asyncio.sleep(0.5)
        
        # Try different argument formats
        test_formats = [
            # Format 1: Direct args
            {"text": "test", "period": "Q1_FY2024"},
            # Format 2: Nested in arguments
            {"arguments": {"text": "test", "period": "Q1_FY2024"}},
            # Format 3: Just text
            {"text": "test"},
        ]
        
        for i, args in enumerate(test_formats):
            print(f"\n=== Test {i+1}: {args} ===")
            tool = {"jsonrpc": "2.0", "id": i+1, "method": "tools/call",
                    "params": {"name": "extract_financial_entities", "arguments": args}}
            response = await client.post(url, json=tool, 
                                        headers={"Accept": "application/json, text/event-stream",
                                                "mcp-session-id": session_id})
            
            # Parse response
            for line in response.text.split('\n'):
                if line.strip().startswith('data: '):
                    data = line.strip()[6:]
                    import json
                    result = json.loads(data)
                    if "error" in result:
                        print(f"❌ Error: {result['error']['message']}")
                    else:
                        print(f"✅ Success!")
                        break

asyncio.run(test())