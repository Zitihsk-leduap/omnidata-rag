from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.pipeline import query_rag, get_gemini_model, get_reranker, get_hybrid_retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model/reranker/vector-index caches once at startup so the
    # first real request isn't the one paying for cold model loads.
    get_gemini_model()
    get_reranker()
    get_hybrid_retriever()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


class ChatMessage(BaseModel):
    message: str


GREETINGS = {"hi", "hello", "hey", "greetings"}

CANNED_RESPONSES = {
    "how can you help": "I can help you with information, answering questions, and providing assistance on various topics.",
    "who are you": "I am an AI-powered chatbot designed to assist you with your queries.",
    "what is your purpose": "My purpose is to provide helpful and accurate information to users like you.",
}


def chatbot_answer(user_input: str) -> dict:
    text = user_input.lower().strip()

    if text in GREETINGS:
        return {"answer": "Hello! How can I assist you today?", "sources": []}

    for phrase, response in CANNED_RESPONSES.items():
        if phrase in text:
            return {"answer": response, "sources": []}

    result = query_rag(user_input)
    return {"answer": result["answer"], "sources": result["sources"]}


@app.post("/chat")
def chat(msg: ChatMessage):
    result = chatbot_answer(msg.message)
    return {"reply": result["answer"], "sources": result["sources"], "status": "success"}
