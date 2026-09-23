import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from intent_controller import StreamIntentController
from search_engine import hybrid_search

app = FastAPI()
controller = StreamIntentController()

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("UI connected to streaming socket.")
    
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            transcript = payload.get("transcript", "")

            # Evaluate streaming action using intent controller
            decision = controller.analyze_stream(transcript)

            if decision["action"] == "WAIT":
                await websocket.send_json({
                    "status": "WAITING",
                    "reason": decision["reason"],
                    "transcript": transcript,
                    "retrieved_docs": []
                })

            elif decision["action"] == "SUPPRESS":
                await websocket.send_json({
                    "status": "SUPPRESSED",
                    "reason": decision["reason"],
                    "transcript": transcript,
                    "retrieved_docs": []
                })

            elif decision["action"] == "RETRIEVE":
                all_results = []
                for intent in decision["intents"]:
                    results = hybrid_search(intent, top_k=2)
                    all_results.extend(results)

                await websocket.send_json({
                    "status": "RETRIEVED",
                    "reason": decision["reason"],
                    "intents": decision["intents"],
                    "transcript": transcript,
                    "retrieved_docs": all_results
                })

    except WebSocketDisconnect:
        print("UI disconnected.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)