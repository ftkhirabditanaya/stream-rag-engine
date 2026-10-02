import streamlit as st
import json
import asyncio
import websockets
import time

# Page config for wide layout and dark high-tech theme
st.set_page_config(
    page_title="Streaming Live RAG Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-Tech Glassmorphism CSS Injection
st.markdown("""
<style>
    /* Dark space mesh grid background styling */
    .stApp {
        background-color: #060913;
        background-image: 
            radial-gradient(at 10% 10%, rgba(0, 240, 255, 0.08) 0px, transparent 50%),
            radial-gradient(at 90% 10%, rgba(168, 85, 247, 0.08) 0px, transparent 50%),
            radial-gradient(#1e293b 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 28px 28px;
        color: #f8fafc;
    }
    
    /* Custom HUD Header */
    .hud-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.2rem;
        background: linear-gradient(135deg, #00f0ff 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    
    /* Live status badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 6px 14px;
        border-radius: 20px;
        color: #34d399;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
        font-family: monospace;
    }
    
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #34d399;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px #34d399;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.4; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.2); }
        100% { opacity: 0.4; transform: scale(0.9); }
    }

    /* Entity and graph pills */
    .entity-pill {
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-family: monospace;
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }

    .entity-pill-new {
        background: rgba(16, 185, 129, 0.18);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #34d399;
        box-shadow: 0 0 8px rgba(16, 185, 129, 0.3);
    }

    /* Telemetry cards */
    .telemetry-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(16px);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    
    .citation-pill {
        background: rgba(168, 85, 247, 0.2);
        border: 1px solid rgba(168, 85, 247, 0.5);
        color: #e9d5ff;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-family: monospace;
        display: inline-block;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# HUD Header
st.markdown('<div class="hud-title">⚡ STREAMING LIVE RAG ENGINE</div>', unsafe_allow_html=True)
st.markdown("""
<div class="status-badge">
    <div class="pulse-dot"></div> ENGINE ONLINE | WEBSOCKET: WS://LOCALHOST:8000/WS/STREAM
</div>
""", unsafe_allow_html=True)

# Sidebar Control
with st.sidebar:
    st.header("⚙️ Session Control")
    st.write("Manage active state versioning & knowledge graph state.")
    reset_btn = st.button("🔄 Reset Session State (v1)", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🌐 Web App Interface")
    st.markdown("A full-featured standalone Glassmorphism web interface is served directly at:")
    st.code("http://localhost:8000", language="text")

# Main UI split layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("🎙️ Live Speech Transcript Stream")
    
    # Preset triggers for rapid testing
    preset = st.selectbox(
        "Select sample transcript scenario:",
        [
            "What is the venue capacity and what is the cancellation fee?",
            "What is the main event venue capacity in Pune",
            "Please summarize in bullet points",
            "All hackathon project submissions deadline",
            "Type custom prompt..."
        ]
    )
    
    default_text = "" if preset == "Type custom prompt..." else preset
    user_input = st.text_area("Live Transcript Buffer:", value=default_text, height=120)
    
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        simulate_btn = st.button("🚀 Stream Token Payload", use_container_width=True)

with col_right:
    st.subheader("🔍 Real-time Telemetry & Citations")
    telemetry_placeholder = st.empty()

# Helper function to communicate with WebSocket
async def send_ws_payload(payload_dict):
    uri = "ws://localhost:8000/ws/stream"
    start_time = time.time()
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(payload_dict))
            response = await websocket.recv()
            latency = round((time.time() - start_time) * 1000, 2)
            data = json.loads(response)
            data["latency"] = latency
            return data
    except Exception as e:
        return {"error": str(e)}

# Reset logic
if reset_btn:
    res = asyncio.run(send_ws_payload({"command": "reset"}))
    if "error" in res:
        st.sidebar.error(f"Reset Failed: {res['error']}")
    else:
        st.sidebar.success(f"State Differ Reset to {res.get('version', 'v1')}")

# Execute WebSocket payload on button click
if simulate_btn and user_input:
    data = asyncio.run(send_ws_payload({"transcript": user_input}))
    
    with telemetry_placeholder.container():
        if "error" in data:
            st.error(f"WebSocket Connection Failed: {data['error']}. Ensure `python app.py` is running.")
        else:
            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            action_val = data.get("action", "WAIT")
            m1.metric("Action Gate", action_val)
            m2.metric("State Version", data.get("version", "v1"))
            m3.metric("Gate Policy", data.get("gate", "G1_WAIT"))
            m4.metric("P95 Latency", f"{data.get('latency', 0)} ms")
            
            st.markdown("---")
            
            # Reason & Intent decomposition
            st.markdown(f"**Gate Reasoning:** `{data.get('reason', 'N/A')}`")
            
            intents = data.get("intents", [])
            delta_intents = data.get("delta_intents", [])
            if intents:
                st.markdown("**Decomposed Intents & State Delta (Δ):**")
                for idx, intent in enumerate(intents, 1):
                    is_delta = intent in delta_intents
                    status_str = "⚡ [Δ Fetch Executed]" if is_delta else "💤 [Cached / In Context]"
                    st.write(f"- Sub-query {idx}: `{intent}` — **{status_str}**")
            
            # Active entities & Checksum
            entities = data.get("active_entities", [])
            added_entities = data.get("added_entities", [])
            checksum = data.get("graph_checksum", "")
            if entities or checksum:
                st.markdown("**Active Session Knowledge Graph State:**")
                st.markdown(f"Checksum: `{checksum}`")
                entity_html = "".join([
                    f'<span class="entity-pill {"entity-pill-new" if e in added_entities else ""}">{("Δ " if e in added_entities else "") + e}</span>' 
                    for e in entities
                ])
                st.markdown(entity_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            # Retrieved Citations
            docs = data.get("retrieved_docs", [])
            if docs:
                st.markdown("**Grounded Context Citations (RRF Dense + BM25):**")
                for doc in docs:
                    citation = doc.get('citation', '[Doc_Ref]')
                    text = doc.get('text', '')
                    score = doc.get('score', 0.0)
                    st.markdown(f"""
                    <div class="telemetry-card">
                        <span class="citation-pill">{citation}</span> <b>RRF Score: {score:.4f}</b>
                        <p style="margin-top: 8px; color: #cbd5e1; font-size: 0.9rem;">{text}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No vector search fired for current turn (Bypassed by Gate / State Diffing).")