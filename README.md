# ⬡ Hex MCTS & Hybrid PyTorch MuZero Engine

[![PLAY NOW — Live PWA Demo](https://img.shields.io/badge/PLAY_NOW-Live_PWA_Demo-00FFFF?style=for-the-badge&logo=google-chrome&logoColor=0a0e1a)](https://tryggth.github.io/hex/)
[![DOCUMENTATION — PDF Paper](https://img.shields.io/badge/DOCUMENTATION-PDF_Paper-FF4444?style=for-the-badge&logo=adobe-acrobat-reader&logoColor=white)](paper/mcts_hex_paper.pdf)
[![Deploy to GitHub Pages](https://github.com/tryggth/hex/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/tryggth/hex/actions/workflows/deploy-pages.yml)
[![PWA Self-Updating](https://img.shields.io/badge/PWA-Self--Updating-00FFFF?style=flat-square)](https://tryggth.github.io/hex/)

An installable, self-updating **Progressive Web App (PWA)** demonstrating real-time **Monte Carlo Tree Search (MCTS)** and a **Hybrid PyTorch MuZero Neural Engine** for the classic two-player strategy game **Hex**.

🎮 **Live Web Application**: [https://tryggth.github.io/hex/](https://tryggth.github.io/hex/)  
📄 **Mathematical Paper (PDF)**: [**`paper/mcts_hex_paper.pdf`**](paper/mcts_hex_paper.pdf) — *"Monte Carlo Tree Search and Hex: A Topological and Probabilistic Exploration"*

---

## 🌟 Architecture & Dual-Mode System

This application operates in two distinct operational regimes:

1. **Offline Client-Side PWA (Vanilla JavaScript)**:
   - Runs 100% locally inside the browser.
   - Powered by a pure JS Monte Carlo Tree Search (MCTS) engine.
   - Uses an $O(\alpha(V))$ **Disjoint-Set (Union-Find)** data structure with path compression and rank optimization for instant win detection.
   - Executes stochastic playout rollouts via the **Nash Trick**.
2. **Hybrid PyTorch MuZero Backend (FastAPI + WebSockets)**:
   - Offloads AI evaluation to a local Python backend running deep PyTorch neural networks ($h_\theta, g_\theta, f_\theta$).
   - Executes **Latent MCTS** entirely inside learned spatial embeddings without querying game rules or environment transition functions.
   - Streams real-time MCTS search heatmaps and visit counts over WebSockets (`ws://localhost:8000/ws/muzero`).
   - Seamlessly auto-connects when the server is active, with fallback to client-side JS MCTS when offline.

---

## 💻 Setup & Environment

### Prerequisites
- Python 3.10+
- Virtual environment (`venv`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tryggth/hex.git
   cd hex
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required backend dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

---

## 🏋️ Model Self-Play Training Instructions

`backend/train.py` provides full command-line parameterization (`argparse`), an **Experience Replay Buffer**, **Dirichlet Noise Injection** ($\alpha = 0.3$), and **Temperature Sampling** ($\tau = 1.0 \to 0$) for self-play reinforcement learning.

### Quick Test Run (5x5 Board)
```bash
python backend/train.py --board-size 5 --num-games 20 --sims-per-move 100
```

### Overnight Grandmaster Run (7x7 Board)
```bash
python backend/train.py --board-size 7 --num-games 400 --sims-per-move 400 --num-blocks 8 --latent-channels 96
```

---

## 🚀 Running the Live Hybrid Server

1. Launch the FastAPI server:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

2. Open your browser to:
   ```text
   http://localhost:8000
   ```

3. The PWA will load and establish a live WebSocket link to `/ws/muzero`. You can now play against the PyTorch MuZero engine in real-time!

---

## 📁 Repository Structure

```
hex/
├── backend/                          # PyTorch MuZero Backend & Environment
│   ├── main.py                       # FastAPI WebSocket server & static file host
│   ├── hex_env.py                    # Gymnasium-style Hex environment & Union-Find
│   ├── muzero_nets.py                # PyTorch Representation, Dynamics & Prediction nets
│   ├── latent_mcts.py                # Latent space MCTS search engine & PUCT
│   ├── train.py                      # Replay Buffer & Self-Play CLI Trainer
│   ├── requirements.txt              # PyTorch, FastAPI, Uvicorn, WebSockets, tqdm
│   └── model_weights.pth             # Trained PyTorch neural network weights
├── paper/                            # LaTeX Documentation & Research Paper
│   ├── mcts_hex_paper.tex            # LaTeX paper source code
│   └── mcts_hex_paper.pdf            # Compiled PDF paper
├── hex-pwa/                          # Single Page Web Application
│   ├── index.html                    # SPA HTML entry point & UI layout
│   ├── style.css                     # Dark navy/cyan CSS design system
│   ├── app.js                        # Hex engine, MCTS search, WebSocket adapter
│   ├── sw.js                         # Service Worker with auto-update caching
│   └── manifest.json                 # Web App Manifest for PWA installation
└── README.md
```

---

## 📜 License & Credits

- Created for exploring Monte Carlo Tree Search, Topology, and Deep Reinforcement Learning in Hex.
- Hosted on [GitHub Pages](https://tryggth.github.io/hex/).
