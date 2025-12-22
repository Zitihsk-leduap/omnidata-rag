import streamlit as st
import requests

st.set_page_config(page_title="Omnidata RAG Chatbot", layout="wide")

# Custom CSS for WhatsApp-like bubbles with left/right alignment
st.markdown("""
<style>
    /* Container for all messages */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    /* Each message row */
    .chat-row {
        display: flex;
        width: 100%;
    }

    /* Common bubble styling */
    .chat-bubble {
        padding: 12px 18px;
        border-radius: 20px;
        max-width: 60%;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.4;
    }

    /* User message (Right aligned) */
    .user-bubble {
        background-color: #005c4b;
        color: white;
        margin-left: auto; /* Push to right */
        border-bottom-right-radius: 2px;
    }

    /* Bot message (Left aligned) */
    .bot-bubble {
        background-color: #202c33;
        color: #e9edef;
        margin-right: auto; /* Push to left */
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
        st.markdown(f'<div class="chat-row"><div class="chat-bubble user-bubble">{content}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-row"><div class="chat-bubble bot-bubble">{content}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Append user message immediately
    st.session_state.messages.append(("User", user_input))
    st.rerun()

# Handle bot response AFTER rerun
if st.session_state.messages and st.session_state.messages[-1][0] == "User":
    last_user_message = st.session_state.messages[-1][1]

    # Create placeholder for bot "typing"
    bot_placeholder = st.empty()
    bot_placeholder.markdown(
        '<div class="chat-row"><div class="chat-bubble bot-bubble">Generating response...</div></div>',
        unsafe_allow_html=True
    )

    # Call backend RAG API
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": last_user_message},
            timeout=500
        ).json()
        bot_reply = response.get("reply", "No response from server.")
    except Exception as e:
        bot_reply = f"Error: {e}"

    # Replace placeholder with final bot response
    bot_placeholder.markdown(
        f'<div class="chat-row"><div class="chat-bubble bot-bubble">{bot_reply}</div></div>',
        unsafe_allow_html=True
    )

    # Append bot message to session state
    st.session_state.messages.append(("Bot", bot_reply))
