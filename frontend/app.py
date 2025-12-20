import streamlit as st
import requests

st.title("Omnidata rag chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []


user_input = st.text_input("You: ")

if st.button("Send") and user_input:
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": user_input}
        ).json()

        st.session_state.messages.append(("User", user_input))
        st.session_state.messages.append(("Bot", response["reply"]))
    except Exception as e:
        st.error(f"Error: {e}")


# Display chat history
for sender, message in st.session_state.messages:
    if sender == "User":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")