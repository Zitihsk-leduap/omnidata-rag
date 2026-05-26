from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from AI.query import query_rag


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
def health():
    return {"status": "ok"}


class ChatMessage(BaseModel):
    message: str



# Chatbot routing function

def chatbot_answer(user_input:set):
    text_from_user = user_input.lower().strip()

    # Handling Normal questions like greetings

    greetings = ["hi", "hello", "hey", "greetings"]
    if text_from_user in greetings:
        return "Hello! How can I assist you today?"

    if "How can you help" in text_from_user:
        return "I can help you with information, answering questions, and providing assistance on various topics."
    
    if "who are you" in text_from_user:
        return "I am an AI-powered chatbot designed to assist you with your queries."

    if "what is your purpose" in text_from_user:
        return "My purpose is to provide helpful and accurate information to users like you."

    
    return query_rag(user_input)


# Endpoint for chat

@app.post("/chat")
def chat(msg:ChatMessage):
    reply = chatbot_answer(msg.message)
    return {"reply": reply}