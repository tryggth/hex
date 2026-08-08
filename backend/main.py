import os
import json
import random
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Hex MuZero Mock Backend")

# Locate static frontend directory (hex-pwa directory containing index.html)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "hex-pwa"))
if not os.path.exists(STATIC_DIR):
    STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

print(f"[Backend] Hosting frontend PWA from: {STATIC_DIR}")

@app.websocket("/ws/muzero")
async def websocket_muzero(websocket: WebSocket):
    await websocket.accept()
    print("[Backend] WebSocket client connected to /ws/muzero")

    think_task = None
    stop_event = asyncio.Event()

    async def run_simulated_search(board, size, time_limit):
        empty_cells = [i for i, val in enumerate(board) if val == 0]
        if not empty_cells:
            await websocket.send_json({"type": "final_move", "move": -1})
            return

        mock_visits = {m: 0 for m in empty_cells}
        # Choose 2-3 favorite candidate moves for realistic visit distribution
        favored_count = min(3, len(empty_cells))
        favored_moves = random.sample(empty_cells, favored_count)
        total_nodes = 0

        # Simulate 50 search iterations (~1.5s duration)
        for loop in range(1, 51):
            if stop_event.is_set():
                print("[Backend] Search interrupted by stop action")
                break

            await asyncio.sleep(0.03)

            total_nodes += random.randint(20, 50)
            for m in empty_cells:
                inc = random.randint(1, 6)
                if m in favored_moves:
                    inc += random.randint(10, 30)
                mock_visits[m] += inc

            # Send heatmap update every 5 loops or on final loop
            if loop % 5 == 0 or loop == 50:
                visits_list = [{"move": m, "visits": v} for m, v in mock_visits.items()]
                try:
                    await websocket.send_json({
                        "type": "heatmap_update",
                        "total_nodes": total_nodes,
                        "visits": visits_list
                    })
                except Exception as e:
                    print(f"[Backend] Error sending heatmap update: {e}")
                    break

        # Select move with highest mock visit count
        best_move = max(mock_visits.keys(), key=lambda m: mock_visits[m]) if mock_visits else -1
        print(f"[Backend] Completed search. Selected move index: {best_move}")
        try:
            await websocket.send_json({"type": "final_move", "move": best_move})
        except Exception as e:
            print(f"[Backend] Error sending final move: {e}")

    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)
            action = data.get("action")

            if action == "think":
                if think_task and not think_task.done():
                    think_task.cancel()
                stop_event.clear()
                
                board = data.get("board", [])
                size = data.get("size", 7)
                time_limit = data.get("timeLimit", 1500)
                print(f"[Backend] Initiating search on {size}x{size} board ({len(board)} cells)")

                think_task = asyncio.create_task(
                    run_simulated_search(board, size, time_limit)
                )

            elif action == "stop":
                print("[Backend] Received 'stop' command from client")
                stop_event.set()

    except WebSocketDisconnect:
        print("[Backend] WebSocket client disconnected")
    except Exception as e:
        print(f"[Backend] WebSocket exception: {e}")
    finally:
        if think_task and not think_task.done():
            think_task.cancel()

# Mount parent frontend directory at root '/' so FastAPI serves index.html and PWA assets
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
