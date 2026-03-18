
import requests

payload = {
    "jsonrpc": "2.0",
    "method": "add",
    "params": [10, 20],
    "id": 2
}

response = requests.post("http://localhost:4000", json=payload)
print("Client 2 result:", response.json())