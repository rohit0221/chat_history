import requests

BASE_URL = "http://localhost:8000"

chat_payload = {
    "user_id": "user_test",
    "session_id": "session_123",
    "message": "What is the process for a refund?"
}

response = requests.post(f"{BASE_URL}/chatbot", json=chat_payload)

print("Chatbot Response:", response.json())
