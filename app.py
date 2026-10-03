import streamlit as st
import time
from rag_chain import ask_question, llm

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="AI Tech Support Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# GLOBAL STYLES (PRODUCTION UI)
# =====================================================
st.markdown("""
<style>

/* ---------- App Background ---------- */
.stApp {
    background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
    color: #ffffff;
    font-family: 'Inter', sans-serif;
}

/* ---------- Chat Messages ---------- */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.09);
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 14px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}

/* User bubble */
[data-testid="stChatMessage"][aria-label="user"] {
    border-left: 5px solid #00e5ff;
}

/* Assistant bubble */
[data-testid="stChatMessage"][aria-label="assistant"] {
    border-left: 5px solid #7cff00;
}

/* ---------- Chat Input ---------- */
.stChatInput textarea {
    border-radius: 14px;
    background-color: rgba(255,255,255,0.12);
    color: white;
    border: 1px solid rgba(255,255,255,0.2);
}

/* ---------- Badges ---------- */
.badge {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    background: rgba(255,255,255,0.12);
    margin-right: 8px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2027, #203a43);
}

/* ---------- Hide Footer ---------- */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

if llm is None:
    st.warning(
        "Missing Groq API key. Add GROQ_API_KEY in Streamlit Cloud secrets or set it in the environment before starting the app."
    )
    st.stop()

# =====================================================
# HEADER
# =====================================================
st.title("🧠 Intelligent AI Tech Support Assistant")
st.caption("Conversational • Memory-Aware • Hybrid RAG • Production-Ready")

st.markdown("""
<span class="badge">🧠 Memory Aware</span>
<span class="badge">📚 Hybrid RAG</span>
<span class="badge">⚡ Groq Powered</span>
<span class="badge">🔐 Context Safe</span>
""", unsafe_allow_html=True)

st.divider()

# =====================================================
# SIDEBAR (CONTROL PANEL)
# =====================================================
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")

    st.markdown("**Assistant Mode**")
    st.info("AI Tech Support (Production)")

    st.markdown("**System Status**")
    st.success("Memory Enabled")
    st.success("Context Tracking Active")

    st.divider()

    if st.button("🧹 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory = {
            "issue": None,
            "model": None,
            "usage": None,
            "steps_given": []
        }
        st.toast("Conversation cleared", icon="🧹")
        st.rerun()

    st.divider()

    with st.expander("🧠 Memory Inspector (Read-Only)"):
        st.json(st.session_state.get("memory", {}))

# =====================================================
# SESSION STATE
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = {
        "issue": None,
        "model": None,
        "usage": None,
        "steps_given": []
    }

memory = st.session_state.memory

# =====================================================
# METRICS BAR
# =====================================================
col1, col2, col3 = st.columns(3)

col1.metric("Issues Tracked", 1 if memory["issue"] else 0)
col2.metric("Steps Given", len(memory["steps_given"]))
col3.metric("Context Fields", sum(v is not None for v in memory.values()))

st.divider()

# =====================================================
# TYPEWRITER EFFECT
# =====================================================
def type_writer(text, speed=0.015):
    placeholder = st.empty()
    rendered = ""
    for char in text:
        rendered += char
        placeholder.markdown(rendered + "▌")
        time.sleep(speed)
    placeholder.markdown(rendered)

# =====================================================
# WELCOME MESSAGE
# =====================================================
if len(st.session_state.messages) == 0:
    with st.chat_message("assistant"):
        st.markdown("""
👋 **Welcome!**

I’m your **AI Tech Support Assistant**, designed for real-world troubleshooting.

### What I can do:
- Diagnose hardware & software issues  
- Remember context across messages  
- Avoid repeating steps  
- Guide you clearly, step by step  

Start by describing your issue 👇
""")

# =====================================================
# CHAT HISTORY
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =====================================================
# CHAT INPUT
# =====================================================
user_input = st.chat_input("Describe your issue...")

if user_input:
    # ---------------- USER MESSAGE ----------------
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    user_lower = user_input.lower()

    # ---------------- MEMORY EXTRACTION (UNCHANGED) ----------------
    if memory["issue"] is None:
        memory["issue"] = user_input
    elif memory["model"] is None and any(b in user_lower for b in ["hp", "dell", "lenovo"]):
        memory["model"] = user_input
    elif memory["usage"] is None and any(
        w in user_lower for w in ["startup", "every", "always", "heavy", "sometimes"]
    ):
        memory["usage"] = user_input

    # ---------------- FINAL QUERY (UNCHANGED) ----------------
    final_query = f"""
Conversation Memory:
Issue: {memory['issue']}
Laptop Model: {memory['model']}
Usage Pattern: {memory['usage']}
Previous Steps: {memory['steps_given']}

User says:
{user_input}

Decide whether to ask a diagnostic question or provide troubleshooting steps.
Do not repeat questions already answered.
"""

    # ---------------- AI RESPONSE ----------------
    with st.spinner("🤖 Analyzing issue..."):
        try:
            response = ask_question(final_query)
            bot_reply = response["answer"]
        except Exception as e:
            bot_reply = f"⚠️ Something went wrong while generating the response: {e}"
            st.error(str(e))

    memory["steps_given"].append(bot_reply)

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    with st.chat_message("assistant"):
        type_writer(bot_reply)