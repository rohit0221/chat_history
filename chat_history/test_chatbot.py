import requests
import uuid

BASE_URL = "http://localhost:8000"

# Generate a random session ID
session_id = str(uuid.uuid4())  # Generates a unique session ID

# User details
user_id = "user_test"

# Loop to keep asking for user input until 'q' or 'quit' is typed
while True:
    message = input("You: ")
    
    # Check if the user wants to quit the session
    if message.lower() in ['q', 'quit']:
        # Send the 'end session' signal to the server
        chat_payload = {
            "user_id": user_id,
            "session_id": session_id,
            "message": "end session"
        }
        response = requests.post(f"{BASE_URL}/chatbot", json=chat_payload)
        print("Chatbot Response:", response.json())
        break
    
    # Otherwise, send the message to the chatbot API
    chat_payload = {
        "user_id": user_id,
        "session_id": session_id,
        "message": message
    }

    response = requests.post(f"{BASE_URL}/chatbot", json=chat_payload)
    
    print("Chatbot Response:", response.json())
