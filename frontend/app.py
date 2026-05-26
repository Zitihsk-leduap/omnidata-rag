import os
import re
import requests
import streamlit as st

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8080")

st.set_page_config(
    page_title="Nepali RAG Assistant",
    page_icon="📄",
    layout="centered"
)

# ─────────────────────────────────────────────
# CLEAN RESPONSE
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"\[From context:.*?\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap');

/* ── GLOBAL RESET ── */
html, body, .stApp, [data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
    color: #111111 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu, footer {
    display: none !important;
}

.block-container {
    max-width: 780px !important;
    padding: 2.5rem 1.5rem 4rem !important;
}

/* ── HEADER ── */
.rag-header {
    text-align: center;
    padding: 2rem 0 1.5rem;
}
.rag-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    font-weight: 400;
    color: #111;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.rag-header p {
    font-size: 0.95rem;
    color: #6b7280;
    margin: 0;
}

/* ── DIVIDER ── */
.rag-divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 1.5rem 0;
}

/* ── USER BUBBLE ── */
.user-bubble {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
}
.user-bubble-inner {
    background: #111111;
    color: #ffffff !important;
    padding: 10px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 0.93rem;
    line-height: 1.5;
}

/* ── BOT BUBBLE ── */
.bot-bubble {
    display: flex;
    justify-content: flex-start;
    margin: 12px 0;
}
.bot-bubble-inner {
    background: #f8f9fa;
    color: #111111 !important;
    padding: 14px 18px;
    border-radius: 4px 18px 18px 18px;
    max-width: 88%;
    font-size: 0.93rem;
    line-height: 1.7;
    border: 1px solid #e9ecef;
}

/* ── SOURCE CARD ── */
.source-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 3px solid #111;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 8px;
    font-size: 0.82rem;
    color: #374151;
    line-height: 1.5;
}
.source-card b {
    color: #111;
    display: block;
    margin-bottom: 2px;
}

/* ── TEXT AREA ── */
.stTextArea label {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #111111 !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 12px !important;
    padding: 12px 14px !important;
    font-size: 0.95rem !important;
    font-family: 'DM Sans', sans-serif !important;
    caret-color: #111 !important;
    box-shadow: none !important;
    transition: border-color 0.2s;
}
.stTextArea textarea:focus {
    border-color: #111111 !important;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.06) !important;
}
.stTextArea textarea::placeholder {
    color: #9ca3af !important;
}

/* ── BUTTONS ── */
.stButton > button {
    background-color: #111111 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 20px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background-color 0.2s, transform 0.1s !important;
}
.stButton > button:hover {
    background-color: #333333 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Clear button variant */
div[data-testid="column"]:nth-child(2) .stButton > button {
    background-color: #ffffff !important;
    color: #111111 !important;
    border: 1.5px solid #d1d5db !important;
}
div[data-testid="column"]:nth-child(2) .stButton > button:hover {
    background-color: #f3f4f6 !important;
    transform: translateY(-1px) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    background: #ffffff !important;
    margin-top: 6px !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    color: #374151 !important;
    padding: 8px 12px !important;
}

/* ── EMPTY STATE ── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: #9ca3af;
}
.empty-state .icon {
    font-size: 2.5rem;
    margin-bottom: 12px;
}
.empty-state p {
    font-size: 0.9rem;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f1f1f1; }
::-webkit-scrollbar-thumb { background: #ccc; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <h1>📄 Nepali RAG Assistant</h1>
    <p>Ask questions from your documents — in Nepali or English</p>
</div>
<hr class="rag-divider">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─────────────────────────────────────────────
# CHAT HISTORY
# ─────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">💬</div>
        <p>No conversation yet. Ask something below to get started.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="user-bubble">
                <div class="user-bubble-inner">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-bubble">
                <div class="bot-bubble-inner">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)

            if msg.get("sources"):
                with st.expander("📚 View Sources"):
                    for i, s in enumerate(msg["sources"]):
                        st.markdown(f"""
                        <div class="source-card">
                            <b>Source {i+1}</b>
                            {s.get('text', '')}
                        </div>
                        """, unsafe_allow_html=True)

st.markdown("<hr class='rag-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────────
query = st.text_area(
    "Your Question",
    placeholder="Type your question in Nepali or English...",
    height=100,
    label_visibility="visible"
)

col1, col2 = st.columns([3, 1])
ask = col1.button("🔍 Ask")
clear = col2.button("🧹 Clear")

# ─────────────────────────────────────────────
# CLEAR
# ─────────────────────────────────────────────
if clear:
    st.session_state.messages = []
    st.rerun()

# ─────────────────────────────────────────────
# BACKEND CALL
# ─────────────────────────────────────────────
if ask and query.strip():
    st.session_state.messages.append({
        "role": "user",
        "content": query.strip()
    })

    payload = {
        "message": query.strip(),
        "top_k_retrieval": 20,
        "top_k_context": 5
    }

    with st.spinner("Searching documents..."):
        try:
            resp = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=120)

            if resp.status_code == 200:
                data = resp.json()
                answer = clean_text(data.get("reply", "No response received."))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": data.get("sources", [])
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Server error ({resp.status_code}): {resp.text}",
                    "sources": []
                })

        except requests.exceptions.ConnectionError:
            st.session_state.messages.append({
                "role": "assistant",
                "content": "❌ Could not reach the backend. Make sure the server is running.",
                "sources": []
            })

    st.rerun()