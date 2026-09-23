import streamlit as st
import websocket
import json

st.set_page_config(page_title="Samsung PRISM - StreamRAG Engine", layout="wide")

st.title("⚡ Streaming Live RAG Engine")
st.caption("Theme 4: Real-time speculative retrieval, multi-intent search, and query suppression")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎙️ Live Speech Transcript Stream")
    sample_queries = [
        "I want to know about the",
        "What is the venue capacity and what is the cancellation fee?",
        "Format that as 2 bullet points"
    ]
    
    selected = st.selectbox("Select or type live transcript:", sample_queries)
    user_input = st.text_area("Current Live Transcript Buffer:", value=selected, height=100)
    
    send_btn = st.button("Simulate Live Stream Token", type="primary")

with col2:
    st.subheader("🔍 Engine Telemetry & Grounded Citations")
    
    if send_btn and user_input:
        try:
            ws = websocket.create_connection("ws://127.0.0.1:8000/ws/stream")
            ws.send(json.dumps({"transcript": user_input}))
            response = json.loads(ws.recv())
            ws.close()

            status = response.get("status")
            
            if status == "WAITING":
                st.warning(f"⏸️ **Status:** WAITING\n\n**Reason:** {response.get('reason')}")
                
            elif status == "SUPPRESSED":
                st.info(f"🚫 **Status:** SUPPRESSED (Gate G4)\n\n**Reason:** {response.get('reason')}")
                
            elif status == "RETRIEVED":
                st.success(f"⚡ **Status:** RETRIEVED (Gate G2 & G3)\n\n**Reason:** {response.get('reason')}")
                
                st.markdown("### Processed Sub-Intents:")
                for intent in response.get("intents", []):
                    st.code(intent, language="text")

                st.markdown("### Grounded Document Hits (Gate G6 Citations):")
                for doc in response.get("retrieved_docs", []):
                    st.info(f"**Citation:** {doc['citation']} | **RRF Score:** {doc['score']:.4f}\n\n{doc['text']}")

        except Exception as e:
            st.error(f"Error connecting to FastAPI backend: {e}. Make sure app.py is running on port 8000!")