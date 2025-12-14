# chat_rag.py
from query import query_rag

def chat_loop():
    print("🗨️ Welcome to RAG Chat! Type 'exit' to quit.\n")
    
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye! 👋")
            break

        # Query RAG for response
        response = query_rag(user_input)

        # Add to chat history (optional)
        chat_history.append({"user": user_input, "bot": response})

        # Print bot response
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    chat_loop()
