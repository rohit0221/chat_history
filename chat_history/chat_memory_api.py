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

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to ["http://localhost:3000"] if needed)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

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


import openai

import openai

import openai

import openai

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

    # ✅ Correct OpenAI API call for chat models
    response = openai.chat.completions.create(
        model=model_name,  # The model to use (e.g., gpt-4 or gpt-3.5-turbo)
        messages=messages  # Correct parameter for chat models
    )

    return response.choices[0].message.content.strip()





@app.post("/chatbot")
def chatbot(chat: ChatMessage):
    """Handles user messages, retrieves chat history, and generates AI responses."""
    try:
        session_id = chat.session_id
        user_input = chat.message
        user_id = chat.user_id

        print(f"🔹 Received message: {user_input} (Session: {session_id})")

        # ✅ If session is new, load history from PostgreSQL
        redis_key = f"chat:{session_id}"
        history = retrieve_chat_history(session_id)

        if not history:
            cursor.execute("""
                SELECT role, message FROM chat_messages 
                WHERE session_id = %s ORDER BY created_at ASC
            """, (session_id,))
            history = [f"{row[0]}: {row[1]}" for row in cursor.fetchall()]
            
            # Save history in Redis for faster retrieval
            if history:
                redis_client.rpush(redis_key, *history)
                redis_client.expire(redis_key, 3600)  # 1-hour expiry

        print(f"📜 Loaded Chat History: {history}")

        # ✅ Handle "end session" trigger
        if user_input.lower() in ["end session", "q", "quit"]:
            print(f"📌 Ending session {session_id} and generating summary...")

            # Generate summary using OpenAI
            summary_prompt = f"Summarize the following conversation:\n{history}"
            summary_response = openai.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4"),
                messages=[{"role": "user", "content": summary_prompt}]
            )

            session_summary = summary_response.choices[0].message.content.strip()
            print(f"📜 Generated Summary: {session_summary}")

            # Store summary in PostgreSQL
            cursor.execute("""
                INSERT INTO conversation_summaries (user_id, session_id, summary, created_at)
                VALUES (%s, %s, %s, %s)
            """, (user_id, session_id, session_summary, datetime.now()))
            conn.commit()

            # Remove session data from Redis
            redis_client.delete(redis_key)

            return {"status": "Session ended", "summary": session_summary}

        # ✅ Store user message in PostgreSQL
        cursor.execute("""
            INSERT INTO chat_messages (user_id, session_id, role, message) 
            VALUES (%s, %s, %s, %s)
        """, (user_id, session_id, "user", user_input))
        conn.commit()

        # ✅ Find similar past conversations
        similar_chats = find_similar_conversations(user_input)

        # ✅ Generate AI response
        ai_response = generate_chat_response(history, similar_chats, user_input)
        print(f"🤖 AI Response: {ai_response}")

        # ✅ Store AI response in PostgreSQL
        cursor.execute("""
            INSERT INTO chat_messages (user_id, session_id, role, message) 
            VALUES (%s, %s, %s, %s)
        """, (user_id, session_id, "assistant", ai_response))
        conn.commit()

        # ✅ Store both messages in Redis
        redis_client.rpush(redis_key, f"user: {user_input}")
        redis_client.rpush(redis_key, f"assistant: {ai_response}")
        redis_client.expire(redis_key, 3600)  # Extend expiry

        return {"response": ai_response, "chat_history": history}

    except Exception as e:
        print(f"❌ Chatbot API Error: {e}")
        return {"error": "Internal Server Error"}





def generate_summary(messages: List[str]) -> str:
    """Generate a summary of the conversation using OpenAI."""
    summary_prompt = "Please summarize the following conversation:\n" + "\n".join(messages)
    
    response = client.completions.create(
        model=os.environ.get("OPENAI_MODEL"),
        prompt=summary_prompt,
        max_tokens=150
    )
    
    return response.choices[0].text.strip()  # Return the summary text

@app.post("/save_summary/{session_id}")
def save_summary(session_id: str, user_id: str):
    """Generate and save a summary of the conversation."""

    # Retrieve full chat history
    cursor.execute(
        "SELECT message FROM chat_messages WHERE session_id = %s ORDER BY created_at ASC",
        (session_id,),
    )
    messages = [row[0] for row in cursor.fetchall()]

    if not messages:
        return {"status": "No messages found to summarize"}

    try:
        # ✅ Use correct OpenAI chat API call
        summary_prompt = f"Summarize the following conversation in a few sentences:\n{messages}"
        summary_response = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4"),
            messages=[{"role": "user", "content": summary_prompt}]
        )

        session_summary = summary_response.choices[0].message.content.strip()
        print(f"📜 Generated Summary: {session_summary}")

        # ✅ Store the summary in PostgreSQL
        cursor.execute(
            "INSERT INTO conversation_summaries (user_id, session_id, summary, created_at) VALUES (%s, %s, %s, %s)",
            (user_id, session_id, session_summary, datetime.now()),
        )
        conn.commit()

        return {"status": "Session ended", "summary": session_summary}

    except Exception as e:
        print(f"❌ Error in save_summary: {e}")
        return {"error": "Failed to generate summary"}



# 1️⃣ Allow User to Retrieve Past Conversations
# Right now, each session starts fresh. Add an API to list past conversations and retrieve messages from a selected session.

@app.get("/conversations/{user_id}")
def get_conversations(user_id: str):
    """Retrieve conversation summaries for a user."""
    cursor.execute("SELECT session_id, summary, created_at FROM conversation_summaries WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
    conversations = cursor.fetchall()

    return {
        "conversations": [
            {"session_id": row[0], "summary": row[1], "created_at": row[2].isoformat()}
            for row in conversations
        ]
    }

# 2️⃣ Retrieve Full Chat Messages for a Past Session
# Once users see a list of summaries, they might want to click on a session and retrieve full messages.

@app.get("/chat_history/{session_id}")
@app.get("/chat_history/{session_id}")
def get_chat_history(session_id: str):
    """Retrieve full chat history for a given session."""
    cursor.execute("""
        SELECT role, message, created_at FROM chat_messages 
        WHERE session_id = %s ORDER BY created_at ASC
    """, (session_id,))
    
    messages = cursor.fetchall()

    return {
        "session_id": session_id,
        "messages": [
            {"role": row[0], "message": row[1], "created_at": row[2].isoformat()}
            for row in messages
        ]
    }

