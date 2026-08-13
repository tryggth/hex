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

This application operates across two distinct operational regimes:

1. **Offline Client-Side PWA (Vanilla JavaScript)**:
   - Runs 100% locally inside the browser.
   - Powered by a pure JS Monte Carlo Tree Search (MCTS) engine.
   - Uses an $\mathcal{O}(\alpha(V))$ **Disjoint-Set (Union-Find)** data structure with path compression and rank optimization for instant win detection.
   - Executes stochastic playout rollouts via the **Nash Trick**.
2. **Hybrid PyTorch MuZero Backend (FastAPI + WebSockets)**:
   - Offloads AI evaluation to a local Python backend running deep PyTorch neural networks ($h_\theta, g_\theta, f_\theta$).
   - Executes **Latent MCTS** entirely inside learned spatial embeddings without querying game rules or environment transition functions.
   - Streams real-time MCTS search heatmaps and visit counts over WebSockets (`ws://localhost:8000/ws/muzero`).
   - Seamlessly auto-connects when the server is active, with fallback to client-side JS MCTS when offline.

---

## 🛠️ Core Algorithmic Breakthroughs & Patches

The PyTorch Latent MCTS backend includes critical reinforcement learning enhancements:

1. **Behavioral Cloning (Imitation Learning)**:
   - Bypassed the sparse-reward "cold start" by generating a massive expert dataset using the Classic MCTS engine.
   - Cloned its policy and value targets via BPTT, providing dense and immediate high-quality learning signals.
2. **Backpropagation Through Time (BPTT) Unrolled Training**:
   - Upgraded `ExperienceReplayBuffer` to store and sample full game trajectories.
   - Unrolls trajectories $K=5$ steps into the future using `recurrent_inference()` to train the Dynamics Network ($g_\theta$).
   - Applies gradient scaling ($0.5\times$) on latent states and loss masking on padded steps to prevent gradient explosion.
3. **Monotonic Legal Move Scoping (Inception Bug Fix)**:
   - Tracks node-specific `legal_actions` arrays during tree expansion.
   - Prunes previously played actions down each sub-branch to ensure the latent network never evaluates illegal or overlapping moves.
4. **Reward Pollution Removal**:
   - Uses strict zero-sum minimax value backpropagation (`value = -value`).
   - Ignores intermediate reward predictions during search, aligning with the terminal-only nature of Hex.

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

## 🏋️ Model Training Pipelines

The V4 pipeline uses a robust 3-stage process involving an **Experience Replay Buffer** with BPTT trajectory sampling, **Dirichlet Noise Injection** ($\alpha = 0.3$), and **Temperature Sampling** for both imitation learning and self-play fine-tuning.

### Stage 1: Expert Data Generation
Generate a high-quality dataset using the Classic MCTS engine as the teacher:
```bash
PYTHONPATH=backend python backend/generate_expert_data.py --board-size 7 --num-games 1000 --sims-per-move 1000 --output backend/expert_data_7x7.pkl
```

### Stage 2: Supervised Behavioral Cloning
Train the PyTorch MuZero network to clone the expert's policy and value evaluations:
```bash
PYTHONPATH=backend python backend/train_supervised.py --board-size 7 --dataset backend/expert_data_7x7.pkl --epochs 15 --batch-size 64 --lr 1e-3 --run-id v4_clone
```

### Stage 3: Self-Play Fine-Tuning
Fine-tune the cloned model through zero-knowledge self-play to push beyond the teacher's capabilities:
```bash
PYTHONPATH=backend python -m backend.train --board-size 7 --load-weights backend/runs/v4_clone/model_weights.pth --num-games 500 --sims-per-move 400 --lr 1e-4 --run-id v4_fine_tune
```

---

## 📊 Evaluation & Benchmarking Suite

The project features a dedicated statistical evaluation suite to benchmark the MuZero neural network against classic MCTS engines:

### 1. Statistical Hypothesis Testing (SPRT)
Wald's **Sequential Probability Ratio Test (SPRT)** evaluates whether the trained Latent MCTS agent performs statistically significantly better than a uniform random player (Classic MCTS at 1 simulation).
```bash
PYTHONPATH=backend python -m backend.arena_sprt
```
- **Null Hypothesis ($H_0$)**: Win rate $p \le 0.50$ (No better than random chance).
- **Alternative Hypothesis ($H_1$)**: Win rate $p \ge 0.55$ (Statistically superior).
- Computes real-time Log-Likelihood Ratio (LLR) bounds ($A \approx 2.94, B \approx -2.94$) to reach decision efficiency.

### 2. Logistic CSE Parity Benchmark
Calculates the **Compute Simulation Equivalent (CSE)** (or "Network IQ") by fitting a 4-parameter Sigmoid curve across 8 anchor points ($X \in [1, 5, 10, 25, 50, 100, 200, 400]$ Classic MCTS simulations).
```bash
PYTHONPATH=backend python -m backend.arena_logistic
```
- **Sigmoid Model**: $P(\text{win}) = \frac{1}{1 + e^{-k(\ln(X) - \ln(\text{CSE}))}}$
- **Real-Time ASCII Dashboard**: Displays a split-screen view featuring a live `plotext` regression chart on the left and a colorized 7x7 Hex game board on the right.

### 3. Final V4 Benchmark Results
After executing the 3-stage V4 pipeline, the final agent achieved a **Classic Simulation Equivalent (CSE) of 3,787.1**. This demonstrates that the Latent MCTS effectively compressed nearly 4,000 stochastic rollouts of a Classic MCTS engine into a single 400-node latent tree search, providing a **Wall-Clock Speedup of 0.55x** over the equivalent Classic MCTS engine.

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
│   ├── train.py                      # BPTT Replay Buffer & Self-Play CLI Trainer
│   ├── arena.py                      # Binary search simulation parity benchmarker
│   ├── arena_sprt.py                 # Wald's Sequential Probability Ratio Test
│   ├── arena_logistic.py             # Logistic regression CSE benchmark dashboard
│   ├── requirements.txt              # PyTorch, FastAPI, Uvicorn, WebSockets, plotext, scipy
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
├── SPRT_CSE_SUMMARY.md               # Executive architectural summary report
└── README.md
```

---

## 📜 License & Credits

- Created for exploring Monte Carlo Tree Search, Topology, and Deep Reinforcement Learning in Hex.
- Hosted on [GitHub Pages](https://tryggth.github.io/hex/).
