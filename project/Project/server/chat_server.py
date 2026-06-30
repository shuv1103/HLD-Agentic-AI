from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY_1"))

app = FastAPI()

# Allow requests from Streamlit or your HTML frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store chat sessions
sessions = {}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_msg = data.get("message", "")
    session_id = data.get("session_id", "default")

    if not user_msg:
        return {"reply": "Please enter a message."}

    # Ensure session exists
    if session_id not in sessions:
        sessions[session_id] = []

    # ✅ Gemini expects `parts` instead of `content`
    sessions[session_id].append({"role": "user", "parts": [user_msg]})

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        chat = model.start_chat(history=sessions[session_id])
        response = chat.send_message(user_msg)
        bot_reply = response.text or "(No reply received)"
    except Exception as e:
        print("Error:", e)
        return {"reply": "⚠️ Error connecting to Gemini server."}

    # Save the model’s reply
    sessions[session_id].append({"role": "model", "parts": [bot_reply]})

    return {"reply": bot_reply}

"""
CLI Command to run chatbot
python -m uvicorn server.chat_server:app --reload --port 8000
"""