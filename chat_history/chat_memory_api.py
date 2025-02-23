from fastapi import FastAPI, Depends
from pydantic import BaseModel
import redis
import psycopg2
from datetime import datetime
from typing import List
import numpy as np
import json
import openai
import os
import os
from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key (You can also load from environment variables)
open_api_key=os.environ.get("OPENAI_API_KEY")

app = FastAPI()

# Connect to Redis (Short-Term Memory)
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Connect to PostgreSQL (Long-Term Memory + Vector Search)
conn = psycopg2.connect(
    dbname="chat_memory",
    user="postgres",
    password="password",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# Define Embedding Dimension (OpenAI embeddings are 1536-dimensional)
EMBEDDING_SIZE = 1536

# Pydantic Models
class Message(BaseModel):
    user_id: str
    session_id: str
    role: str  # "user" or "assistant"
    content: str

class SearchRequest(BaseModel):
    query_vector: List[float]

# Store messages in Redis (Short-Term Memory)
@app.post("/chat")
def store_message(message: Message):
    redis_key = f"chat:{message.session_id}"
    redis_client.rpush(redis_key, f"{message.role}: {message.content}")
    return {"status": "Message stored"}

# Retrieve chat history from Redis
@app.get("/chat/{session_id}")
def get_chat_history(session_id: str):
    redis_key = f"chat:{session_id}"
    messages = redis_client.lrange(redis_key, 0, -1)  # Get all messages
    return {"messages": messages}

# Function to summarize conversation (Placeholder for AI model)
def generate_summary(messages):
    return "This is a placeholder summary of the chat."



def generate_openai_embedding(text: str):
    """Fetch OpenAI embeddings for the given text using OpenAI's latest API format."""
    client = openai.OpenAI()  # ✅ Create an OpenAI client

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]  # OpenAI now requires a list input
    )
    return response.data[0].embedding  # ✅ Correct way to access the embedding


# Move chat from Redis to PostgreSQL and generate embeddings
@app.post("/save_summary/{session_id}")
def save_summary(session_id: str, user_id: str):
    redis_key = f"chat:{session_id}"
    messages = redis_client.lrange(redis_key, 0, -1)  # Retrieve all messages
    if not messages:
        return {"status": "No messages found"}

    summary = generate_summary(messages)  # Replace with real AI summarization
    
    # 🔹 **Get OpenAI Embedding Instead of Random Vector**
    embedding_vector = generate_openai_embedding(summary)

    cursor.execute("""
        INSERT INTO conversation_summaries (user_id, session_id, summary, embedding, created_at) 
        VALUES (%s, %s, %s, %s, %s)
    """, (user_id, session_id, summary, embedding_vector, datetime.now()))
    
    conn.commit()
    
    # Clear Redis chat history after storing summary
    redis_client.delete(redis_key)
    
    return {"status": "Summary stored", "summary": summary}

# Retrieve conversation summaries from PostgreSQL
@app.get("/conversations/{user_id}")
def get_conversations(user_id: str):
    cursor.execute("SELECT session_id, summary, created_at FROM conversation_summaries WHERE user_id = %s", (user_id,))
    conversations = cursor.fetchall()
    
    results = [
        {"session_id": row[0], "summary": row[1], "created_at": row[2].isoformat()}
        for row in conversations
    ]
    
    return {"conversations": results}

# API Endpoint to search similar conversations using `pgvector`
@app.post("/search_similar")
def search_similar(request: SearchRequest):
    query_vector = request.query_vector

    # Ensure query vector is exactly 1536 dimensions
    if len(query_vector) != 1536:
        return {"error": f"Vector must be 1536 dimensions, received {len(query_vector)}"}

    try:
        # Convert list to proper PostgreSQL array format (JSON dump ensures brackets [])
        query_vector_str = json.dumps(query_vector)

        # Debug: Print query string (First 100 chars for sanity check)
        print(f"Query vector: {query_vector_str[:100]}...")

        # Perform similarity search in PostgreSQL
        cursor.execute("""
            SELECT id, user_id, session_id, summary, embedding <=> %s::vector AS distance
            FROM conversation_summaries
            ORDER BY distance
            LIMIT 5;
        """, (query_vector_str,))

        results = cursor.fetchall()

        return {
            "results": [
                {"id": row[0], "user_id": row[1], "session_id": row[2], "summary": row[3], "distance": row[4]}
                for row in results
            ]
        }
    
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return {"error": "Internal Server Error, check logs"}
