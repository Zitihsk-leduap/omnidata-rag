import streamlit as st
import requests

st.set_page_config(page_title="Omnidata RAG Chatbot", layout="wide")

# Custom CSS for WhatsApp-like bubbles
st.markdown("""
<style>
    /* Container for the chat */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    /* Common bubble styling */
    .chat-bubble {
        padding: 12px 18px;
        border-radius: 20px;
        max-width: 70%;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-bottom: 10px;
        line-height: 1.4;
    }
    /* User message (Right aligned) */
    .user-bubble {
        align-self: flex-end;
        background-color: #005c4b; /* WhatsApp dark green */
        color: white;
        border-bottom-right-radius: 2px;
    }
    /* Bot message (Left aligned) */
    .bot-bubble {
        align-self: flex-start;
        background-color: #202c33; /* WhatsApp dark gray/blue */
        color: #e9edef;
        border-bottom-left-radius: 2px;
    }
    /* Remove default Streamlit padding */
    .stChatMessage {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Omnidata RAG Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages using custom HTML/CSS
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, content in st.session_state.messages:
    if role == "User":
        st.markdown(f'<div class="chat-bubble user-bubble">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble bot-bubble">{content}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Append user message
    st.session_state.messages.append(("User", user_input))
    
    # Call backend RAG API
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": user_input}
        ).json()
        bot_reply = response.get("reply", "No response from server.")
    except Exception as e:
        bot_reply = f"Error: {e}"

    # Append bot response
    st.session_state.messages.append(("Bot", bot_reply))
    
    # Rerun to update the UI with new messages
    st.rerun()