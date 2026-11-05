import os
import sys
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(
    page_title="RoboChat - ROS2 TA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A1128 !important;
    color: #E6E6FA !important;
}
[data-testid="stSidebar"] {
    background-color: #0F1D38 !important;
    border-right: 2px solid #7C3AED;
}
[data-testid="stHeader"] { 
    background: transparent; 
}

.stMarkdown p, .stMarkdown li, .stMarkdown span:not(pre span):not(code span),
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span:not(pre span):not(code span) {
    color: #E6E6FA !important;
}

h1, h2, h3, h4, h5, h6, .st-emotion-cache-10trblm { 
    color: #F59E0B !important; 
}
label, .stTextInput label, .stSelectbox label { 
    color: #E6E6FA !important; 
    font-weight: 500;
}

[data-testid="stChatMessage"] {
    background-color: #132247 !important;
    border-radius: 10px !important;
    border-left: 4px solid #7C3AED;
    margin-bottom: 12px;
    padding: 14px 18px !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    border-left-color: #F59E0B !important;
    background-color: #172A54 !important;
}

code:not(pre code) {
    background-color: #1E1B4B !important;
    color: #FCD34D !important;
    padding: 2px 7px !important;
    border-radius: 4px !important;
    border: 1px solid #4338CA !important;
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-size: 0.88em !important;
}

pre, pre code, [data-testid="stCodeBlock"], [data-testid="stCodeBlock"] pre, [data-testid="stCodeBlock"] code, .stCodeBlock {
    background-color: #050B17 !important;
    color: #E2E8F0 !important;
    border: 1px solid #2D3A58 !important;
    border-radius: 8px !important;
    font-family: 'Consolas', 'Fira Code', 'Courier New', monospace !important;
    font-size: 0.9em !important;
}

[data-testid="stCodeBlock"] pre {
    background-color: #050B17 !important;
    padding: 14px 18px !important;
}

pre code span, [data-testid="stCodeBlock"] span {
    background: transparent !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #132247 !important;
    color: #E6E6FA !important;
    border: 1.5px solid #7C3AED !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #A5B4FC !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #F59E0B !important;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.3) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #5B21B6) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { 
    opacity: 0.9; 
}

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 16px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-right: 8px;
}
.badge-explain  { background: #5B21B6; color: #EDE9FE; border: 1px solid #8B5CF6; }
.badge-debug    { background: #881337; color: #FFE4E6; border: 1px solid #E11D48; }
.badge-search   { background: #064E3B; color: #D1FAE5; border: 1px solid #10B981; }
.badge-chitchat { background: #1E3A8A; color: #DBEAFE; border: 1px solid #3B82F6; }

.source-card {
    background: #0F1D38;
    border: 1px solid #2A3B5C;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 8px;
    font-size: 0.85rem;
    color: #E6E6FA !important;
}
.source-card strong { 
    color: #F59E0B; 
}

.sidebar-panel {
    background: #132247;
    border: 1px solid #2A3B5C;
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 14px;
}
.sidebar-label { 
    color: #DDD6FE; 
    font-size: 0.8rem; 
    font-weight: 600; 
    text-transform: uppercase; 
    letter-spacing: 0.8px; 
}
.conf-value { 
    color: #F59E0B; 
    font-size: 1.4rem; 
    font-weight: 700; 
}

.stProgress > div > div { 
    background-color: #7C3AED !important; 
}

hr { 
    border-color: #2A3B5C !important; 
}

[data-testid="stExpander"] { 
    background-color: #132247 !important;
    border: 1px solid #2A3B5C !important; 
    border-radius: 8px !important; 
}
[data-testid="stExpander"] details summary {
    color: #F59E0B !important;
}
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {
        "messages": [],
        "last_intent": None,
        "last_confidence": None,
        "last_probs": {},
        "last_sources": [],
        "last_model": "[Ready]",
        "groq_key": os.getenv("GROQ_API_KEY", ""),
        "use_bad_key": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


@st.cache_resource(show_spinner="Loading classifier and vector store...")
def get_router():
    from agents.router import route
    return route


INTENT_META = {
    "Explain_Concept": ("badge-explain",  "Explain Concept"),
    "Debug_Code":      ("badge-debug",    "Debug Code"),
    "Search_Paper":    ("badge-search",   "Search Paper"),
    "Chitchat_Other":  ("badge-chitchat", "Chitchat"),
}

def intent_badge(label: str) -> str:
    cls, display = INTENT_META.get(label, ("badge-chitchat", label))
    return f'<span class="badge {cls}">{display}</span>'


def source_card(src: dict) -> str:
    topic_tag = f'<span style="color:#A78BFA;font-size:0.75rem">[{src["topic"]}]</span> ' if src.get("topic") else ""
    snippet = src.get("snippet", "")
    return (
        f'<div class="source-card">'
        f'{topic_tag}<strong>{src["source"]}</strong><br>'
        f'<span style="color:#D8D8F6">{snippet}</span>'
        f'</div>'
    )


with st.sidebar:
    st.markdown("## RoboChat")
    st.markdown('<p style="color:#DDD6FE;font-size:0.85rem">ROS2 and Robotics TA Assistant</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sidebar-panel">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">Last Routing Decision</p>', unsafe_allow_html=True)
    if st.session_state.last_intent:
        badge_html = intent_badge(st.session_state.last_intent)
        st.markdown(badge_html, unsafe_allow_html=True)

        conf = st.session_state.last_confidence or 0
        st.markdown(f'<p class="conf-value">{conf*100:.1f}%</p>', unsafe_allow_html=True)
        st.markdown('<p class="sidebar-label">Confidence</p>', unsafe_allow_html=True)
        st.progress(conf)

        if st.session_state.last_probs:
            st.markdown('<p class="sidebar-label" style="margin-top:10px;">All Probabilities</p>', unsafe_allow_html=True)
            for lbl, p in sorted(st.session_state.last_probs.items(), key=lambda x: -x[1]):
                bar_width = int(p * 100)
                _, disp = INTENT_META.get(lbl, ("", lbl))
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                    f'<span style="width:125px;font-size:0.78rem;color:#E6E6FA">{disp}</span>'
                    f'<div style="flex:1;height:6px;background:#0A1128;border-radius:3px">'
                    f'<div style="width:{bar_width}%;height:100%;background:#7C3AED;border-radius:3px"></div>'
                    f'</div>'
                    f'<span style="font-size:0.78rem;color:#F59E0B;width:38px;text-align:right;">{p*100:.1f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown('<p style="color:#A5B4FC">Ask a question to see routing</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-panel">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-label">LLM Backend</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:#F59E0B;font-weight:600;margin:4px 0 0 0;">{st.session_state.last_model}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Settings", expanded=False):
        new_key = st.text_input(
            "Groq API Key",
            value=st.session_state.groq_key,
            type="password",
            help="Leave blank to use GROQ_API_KEY from .env",
        )
        if new_key != st.session_state.groq_key:
            st.session_state.groq_key = new_key

        bad_key = st.checkbox(
            "Test Fallback (force bad Groq key)",
            value=st.session_state.use_bad_key,
            help="Sends a deliberately invalid key to Groq -> triggers Ollama fallback",
        )
        if bad_key != st.session_state.use_bad_key:
            st.session_state.use_bad_key = bad_key
            if bad_key:
                st.warning("Bad key active — Ollama fallback will be used")

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.last_intent = None
        st.session_state.last_confidence = None
        st.session_state.last_probs = {}
        st.session_state.last_sources = []
        st.rerun()


st.markdown("# RoboChat")
st.markdown(
    '<p style="color:#DDD6FE;margin-top:-12px">Multi-agent ROS2 and Robotics Teaching Assistant | '
    'Powered by Bi-LSTM classifier and LangChain RAG</p>',
    unsafe_allow_html=True,
)
st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            badge = intent_badge(meta.get("intent", ""))
            model_tag = f'<span style="color:#FFAE42;font-size:0.78rem"> | {meta.get("model","")}</span>'
            st.markdown(f'{badge}{model_tag}', unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta", {}).get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["meta"]["sources"]:
                    st.markdown(source_card(src), unsafe_allow_html=True)


def process_query(user_input: str):
    st.session_state.messages.append({"role": "user", "content": user_input})

    history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            history.append(AIMessage(content=m["content"]))

    groq_key = "INTENTIONALLY_BAD_KEY_12345" if st.session_state.use_bad_key else st.session_state.groq_key

    router = get_router()

    with st.spinner("Thinking..."):
        try:
            result = router(user_input, chat_history=history[-10:], groq_api_key=groq_key)
        except Exception as e:
            result = {
                "answer": f"Error: {e}",
                "sources": [],
                "intent": "Chitchat_Other",
                "confidence": 0.0,
                "probabilities": {},
            }

    if st.session_state.use_bad_key:
        model_used = "Ollama llama3.2 (fallback)"
    else:
        model_used = "Groq (openai/gpt-oss-120b)"

    st.session_state.last_intent = result["intent"]
    st.session_state.last_confidence = result["confidence"]
    st.session_state.last_probs = result.get("probabilities", {})
    st.session_state.last_sources = result.get("sources", [])
    st.session_state.last_model = model_used

    meta = {
        "intent": result["intent"],
        "confidence": result["confidence"],
        "sources": result.get("sources", []),
        "model": model_used,
    }
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "meta": meta,
    })


user_input = st.chat_input("Ask a ROS2 or robotics question...")
if user_input:
    process_query(user_input)
    st.rerun()