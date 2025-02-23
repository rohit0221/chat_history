import openai
import json
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import redis
import psycopg2
from datetime import datetime
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Connect to Redis for chat history
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="chat_memory",
    user="postgres",
    password="password",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

# OpenAI API setup
open_api_key=os.environ.get("OPENAI_API_KEY")
client = openai.OpenAI(api_key=open_api_key)

# Define models
class ChatMessage(BaseModel):
    user_id: str
    session_id: str
    message: str

def generate_openai_embedding(text: str):
    """Get OpenAI embeddings for similarity search."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding  # Returns a 1536-dimension vector

def retrieve_chat_history(session_id: str, limit: int = 10):
    """Retrieve last N messages from Redis."""
    redis_key = f"chat:{session_id}"
    messages = redis_client.lrange(redis_key, -limit, -1)
    return messages or []

def find_similar_conversations(query_text: str):
    """Find similar past conversations from PostgreSQL."""
    query_vector = generate_openai_embedding(query_text)
    query_vector_str = json.dumps(query_vector)

    cursor.execute("""
        SELECT session_id, summary, embedding <=> %s::vector AS distance
        FROM conversation_summaries
        ORDER BY distance
        LIMIT 3;
    """, (query_vector_str,))
    
    return cursor.fetchall()


def generate_chat_response(history: List[str], similar_chats: List[tuple], user_input: str):
    """Generate AI chatbot response using OpenAI."""
    messages = [{"role": "system", "content": "You are an AI assistant helping with chat-based memory retrieval."}]

    # Add past chat history (if available)
    for msg in history:
        messages.append({"role": "user", "content": str(msg)})  # Ensure all messages are strings

    # Properly format similar past conversations
    if similar_chats:
        messages.append({"role": "system", "content": "Here are some similar past conversations that might help you:"})
        for session_id, summary, distance in similar_chats:
            formatted_summary = f"Session: {session_id}, Summary: {summary} (Similarity Score: {distance})"
            messages.append({"role": "system", "content": formatted_summary})  # Ensure this is a string

    # Add user input as the last message
    messages.append({"role": "user", "content": user_input})

    # ✅ Debugging: Print the exact structure of `messages` before sending
    print(f"🔹 Sending Messages to OpenAI:\n{json.dumps(messages, indent=2)}")

    # Get model from environment variable (default to GPT-4 if not set)
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4")

    # Send to OpenAI API
    response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )

    return response.choices[0].message.content



@app.post("/chatbot")
def chatbot(chat: ChatMessage):
    """Main chatbot function that retrieves history, finds similar chats, and generates a response."""
    try:
        session_id = chat.session_id
        user_input = chat.message
        user_id = chat.user_id  # Added user_id from the request

        print(f"🔹 Received message: {user_input} (Session: {session_id})")

        # Step 1: Store user message in PostgreSQL
        cursor.execute("""
            INSERT INTO chat_messages (user_id, session_id, role, message) 
            VALUES (%s, %s, %s, %s)
        """, (user_id, session_id, "user", user_input))
        conn.commit()

        # Retrieve past chat history (from Redis)
        history = retrieve_chat_history(session_id)
        print(f"📜 Chat history: {history}")

        # Find similar past conversations (from PostgreSQL)
        similar_chats = find_similar_conversations(user_input)
        print(f"🔍 Similar chats: {similar_chats}")

        # Generate AI response
        ai_response = generate_chat_response(history, similar_chats, user_input)
        print(f"🤖 AI Response: {ai_response}")

        # Step 2: Store AI response in PostgreSQL
        cursor.execute("""
            INSERT INTO chat_messages (user_id, session_id, role, message) 
            VALUES (%s, %s, %s, %s)
        """, (user_id, session_id, "assistant", ai_response))
        conn.commit()

        # Step 3: Store user input and AI response in Redis (short-term memory)
        redis_key = f"chat:{session_id}"
        redis_client.rpush(redis_key, f"user: {user_input}")
        redis_client.rpush(redis_key, f"assistant: {ai_response}")

        return {"response": ai_response}

    except Exception as e:
        print(f"❌ Chatbot API Error: {e}")
        return {"error": "Internal Server Error"}


