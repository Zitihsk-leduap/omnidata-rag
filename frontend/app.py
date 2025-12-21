import streamlit as st
import requests

st.set_page_config(page_title="Omnidata RAG Chatbot", layout="wide")
st.title("Omnidata RAG Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    role, content = msg
    if role == "User":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        with st.chat_message("assistant"):
            st.markdown(content)

# User input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Append user message
    st.session_state.messages.append(("User", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

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
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
