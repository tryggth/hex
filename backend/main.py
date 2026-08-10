import os
import sys
import json
import math
import asyncio
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from muzero_nets import MuZeroModels
from latent_mcts import LatentMCTS

app = FastAPI(title="Hex PyTorch MuZero Backend")

# Locate static frontend directory (hex-pwa containing index.html)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "hex-pwa"))
if not os.path.exists(STATIC_DIR):
    STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

print(f"[MuZero Backend] Hosting static PWA from: {STATIC_DIR}")

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", type=str, default=None, help="Run ID for versioned weights")
args, _ = parser.parse_known_args()

# Dynamically inspect saved model weights to match exact architecture and board size
if args.run_id:
    WEIGHTS_PATH = os.path.join(BASE_DIR, "runs", args.run_id, "model_weights.pth")
else:
    WEIGHTS_PATH = os.path.join(BASE_DIR, "model_weights.pth")
BOARD_SIZE = 7
ACTION_SPACE_SIZE = 49
LATENT_CHANNELS = 32
NUM_RES_BLOCKS = 2

if os.path.exists(WEIGHTS_PATH):
    try:
        saved_weights = torch.load(WEIGHTS_PATH, map_location="cpu")
        if "prediction.policy_fc.weight" in saved_weights:
            ACTION_SPACE_SIZE = saved_weights["prediction.policy_fc.weight"].shape[0]
            BOARD_SIZE = int(math.sqrt(ACTION_SPACE_SIZE))
        if "representation.conv_init.weight" in saved_weights:
            LATENT_CHANNELS = saved_weights["representation.conv_init.weight"].shape[0]
            NUM_RES_BLOCKS = len([k for k in saved_weights.keys() if "representation.res_blocks" in k and "conv1.weight" in k])
        
        model = MuZeroModels(board_size=BOARD_SIZE, action_space_size=ACTION_SPACE_SIZE, latent_channels=LATENT_CHANNELS, num_res_blocks=NUM_RES_BLOCKS)
        model.load_state_dict(saved_weights)
        print(f"[MuZero Backend] Loaded PyTorch master weights ({BOARD_SIZE}x{BOARD_SIZE}, channels={LATENT_CHANNELS}, res_blocks={NUM_RES_BLOCKS}) from: {WEIGHTS_PATH}")
    except Exception as e:
        print(f"[MuZero Backend] Warning: Could not load weights from {WEIGHTS_PATH}: {e}")
        model = MuZeroModels(board_size=BOARD_SIZE, action_space_size=ACTION_SPACE_SIZE, latent_channels=LATENT_CHANNELS, num_res_blocks=NUM_RES_BLOCKS)
else:
    print("[MuZero Backend] No pre-trained weights found; initializing fresh model.")
    model = MuZeroModels(board_size=BOARD_SIZE, action_space_size=ACTION_SPACE_SIZE, latent_channels=LATENT_CHANNELS, num_res_blocks=NUM_RES_BLOCKS)

model.eval()

@app.websocket("/ws/muzero")
async def websocket_muzero(websocket: WebSocket):
    await websocket.accept()
    print("[MuZero Backend] Client connected to /ws/muzero")

    search_task = None
    stop_event = asyncio.Event()

    async def execute_latent_search(board: list, size: int, time_limit: int):
        legal_actions = [i for i, val in enumerate(board) if val == 0]
        if not legal_actions:
            await websocket.send_json({"type": "final_move", "move": -1})
            return

        # Prepare 3D Observation Tensor (1, 3, size, size)
        grid = np.array(board, dtype=np.uint8).reshape((size, size))
        obs_np = np.zeros((1, 3, size, size), dtype=np.float32)
        obs_np[0, 0] = (grid == 1).astype(np.float32)  # P1 Red stones
        obs_np[0, 1] = (grid == 2).astype(np.float32)  # P2 Blue stones

        # Determine current player turn (Red=1 if even number of placed stones, else Blue=2)
        num_placed = np.count_nonzero(grid)
        current_turn = 1 if (num_placed % 2 == 0) else 2
        obs_np[0, 2] = 1.0 if current_turn == 1 else 0.0

        obs_tensor = torch.tensor(obs_np, dtype=torch.float32)

        # Dynamic model scaling if board size differs from default
        active_model = model
        if size != BOARD_SIZE:
            active_model = MuZeroModels(board_size=size, action_space_size=size*size, latent_channels=LATENT_CHANNELS, num_res_blocks=NUM_RES_BLOCKS)
            active_model.eval()

        mcts_engine = LatentMCTS(model=active_model, c_puct=1.25)

        # Run deep Latent MCTS search with PyTorch neural network evaluations and UI time budget
        root = await mcts_engine.search(
            initial_state_tensor=obs_tensor,
            legal_actions=legal_actions,
            num_simulations=400,
            websocket=websocket,
            stop_event=stop_event,
            time_limit_ms=time_limit
        )

        # Select action with highest visit count
        best_move = -1
        max_visits = -1
        for act, child in root.children.items():
            if child.visit_count > max_visits:
                max_visits = child.visit_count
                best_move = act

        print(f"[MuZero Backend] Deep Latent MCTS completed ({root.visit_count} simulations). Selected move index: {best_move}")
        try:
            await websocket.send_json({"type": "final_move", "move": best_move})
        except Exception as e:
            print(f"[MuZero Backend] Error sending final move: {e}")

    try:
        while True:
            data_text = await websocket.receive_text()
            data = json.loads(data_text)
            action = data.get("action")

            if action == "think":
                if search_task and not search_task.done():
                    search_task.cancel()
                stop_event.clear()

                board = data.get("board", [])
                size = data.get("size", 7)
                time_limit = data.get("timeLimit", 1500)
                print(f"[MuZero Backend] Initiating Deep Latent MCTS on {size}x{size} board")

                search_task = asyncio.create_task(
                    execute_latent_search(board, size, time_limit)
                )

            elif action == "stop":
                print("[MuZero Backend] Received 'stop' command")
                stop_event.set()

    except WebSocketDisconnect:
        print("[MuZero Backend] Client disconnected")
    except Exception as e:
        print(f"[MuZero Backend] Exception: {e}")
    finally:
        if search_task and not search_task.done():
            search_task.cancel()

# Mount frontend static files at '/'
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")
