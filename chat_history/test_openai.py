import requests
import numpy as np

# API Base URL
BASE_URL = "http://localhost:8000"

# Step 1: Store a chat message in Redis
chat_url = f"{BASE_URL}/chat"
message_data = {
    "user_id": "user_test",
    "session_id": "session_test",
    "role": "user",
    "content": "This is a test chat message for OpenAI embeddings."
}
chat_response = requests.post(chat_url, json=message_data)
print("Chat Store Response:", chat_response.json())

# Step 2: Save summary (moves messages from Redis to PostgreSQL)
store_url = f"{BASE_URL}/save_summary/session_test"
store_response = requests.post(store_url, params={"user_id": "user_test"})
print("Store Response:", store_response.json())

# Step 3: Generate a Query Vector (Simulating an OpenAI Embedding)
query_text = "Find similar conversations"
query_embedding = np.random.rand(1536).tolist()  # Replace with real OpenAI embeddings

# Step 4: Search for Similar Conversations
search_url = f"{BASE_URL}/search_similar"
search_response = requests.post(search_url, json={"query_vector": query_embedding})
print("Search Response:", search_response.json())
