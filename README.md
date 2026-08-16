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

The PyTorch Latent MCTS backend includes critical reinforcement learning enhancements across V4 and V5 iterations:

1. **5-Channel Boundary-Padded Tensor Representation (V5)**:
   - Eliminates spatial edge blindness by augmenting the standard 3-channel board observation (P1 stones, P2 stones, current player turn) with two explicit topological boundary indicator planes:
     - **Channel 3 (Red Boundary)**: Active $1.0$ on the top and bottom rows ($r = 0$ and $r = B - 1$).
     - **Channel 4 (Blue Boundary)**: Active $1.0$ on the leftmost and rightmost columns ($c = 0$ and $c = B - 1$).
   - Gives convolutional kernels immediate receptive field access to winning perimeter connections without requiring deep feature coordination across distant board edges.

2. **Fully Convolutional Network (FCN) Prediction Architecture (V5)**:
   - Replaced fixed, parameter-heavy dense fully connected linear layers in the Prediction Network with $1 \times 1$ spatial convolutions (`Conv2d(latent_channels, 1, kernel_size=1)`) and Global Adaptive Average Pooling (`AdaptiveAvgPool2d((1, 1))`).
   - Decouples policy and value estimation from rigid spatial dimensions, drastically improving spatial parameter sharing, regularization, and generalization.

3. **Active Sequential Experiment Design & Fisher Information Sampling (V5)**:
   - Replaced unweighted curve fitting with a Binomial Generalized Linear Model (Logit Link MLE) and Delta Method asymptotic covariance estimation on the Fisher Information inverse matrix $\mathcal{I}^{-1}$.
   - Executes two-phase active exploration:
     - **Seed/Hunt Phase**: Geometric striding ($2.0\times / 0.5\times$) to empirically bracket parity ($w > 0.60$ and $w < 0.40$).
     - **Convergence Phase**: Concentrated Fisher Information sampling targeting $c$- and $D$-optimal support points ($x_0$ at 50%, $x_0 \pm \frac{1.543}{|\beta_1|}$ at 25% each).
   - Automatically halts when the 95% Confidence Interval ratio ($\text{CI}_{\text{upper}} / \text{CI}_{\text{lower}}$) converges to $\le 1.30\times$.

4. **Behavioral Cloning (Imitation Learning) & Distillation**:
   - Bypassed the sparse-reward "cold start" by bootstrapping expert trajectories using high-simulation MCTS teacher play.
   - Cloned policy and value targets via unrolled BPTT distillation, providing dense, immediate learning signals.

5. **Backpropagation Through Time (BPTT) Unrolled Training**:
   - Upgraded `ExperienceReplayBuffer` to store and sample full game trajectories.
   - Unrolls trajectories $K=5$ steps into the future using `recurrent_inference()` to train the Dynamics Network ($g_\theta$).
   - Applies gradient scaling ($0.5\times$) on latent states and loss masking on padded steps to prevent gradient explosion.

6. **Monotonic Legal Move Scoping & Pure Minimax Value Inversion**:
   - Tracks node-specific `legal_actions` masks during tree expansion to guarantee the latent network never searches illegal or overlapping coordinates.
   - Enforces strict zero-sum minimax value backpropagation (`value = -value`) without intermediate reward pollution.

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

## 🏋️ Model Training & Distillation Pipeline

The V5 pipeline incorporates a 3-stage imitation learning, supervised distillation, and self-play fine-tuning workflow:

### Stage 1: Master Dataset Generation
Bootstrap a rich trajectory dataset using self-play from the V4 engine with 5-channel boundary padding:
```bash
PYTHONPATH=. python backend/generate_muzero_data.py --board-size 7 --num-games 1000 --sims-per-move 400 --input-channels 5 --output backend/muzero_data_7x7_5ch.pkl
```

### Stage 2: Supervised BPTT Distillation
Train the V5 FCN network with boundary channels on the augmented dataset:
```bash
PYTHONPATH=. python backend/train_supervised.py --board-size 7 --dataset backend/muzero_data_7x7_5ch.pkl --epochs 15 --batch-size 64 --lr 1e-3 --use-fcn --input-channels 5 --run-id v5_clone
```
*(Mean Policy Loss: 2.881, Mean Value Loss: 0.177)*

### Stage 3: Latent MCTS Self-Play Fine-Tuning
Fine-tune the cloned V5 model through zero-knowledge self-play to push beyond the initial dataset:
```bash
PYTHONPATH=. python -m backend.train --run-id v5_fine_tune --board-size 7 --load-weights backend/runs/v5_clone/model_weights.pth --input-channels 5 --use-fcn --num-blocks 8 --latent-channels 96 --num-games 500 --sims-per-move 400 --lr 1e-4
```

---

## 📊 Evaluation & Empirical Benchmarks

The project features a statistical evaluation suite to benchmark MuZero against classic MCTS:

### 1. Active Sequential Logistic CSE Arena
Calculates the exact **Simulation Compute-Scale Equivalence ($N_{50}$)**—the number of Classic MCTS rollouts required to achieve parity ($50\%$ win rate) against 400 MuZero latent lookaheads:
```bash
PYTHONPATH=. python -m backend.arena_logistic \
  --run-id v5_fine_tune \
  --board-size 7 \
  --muzero-sims 400 \
  --input-channels 5 \
  --use-fcn \
  --adaptive \
  --start-sims 5000 \
  --target-ci-ratio 1.30 \
  --baseline-log-n50 8.24
```

### 2. Statistical Hypothesis Testing (SPRT)
Wald's **Sequential Probability Ratio Test (SPRT)** evaluates whether the trained Latent MCTS agent performs statistically significantly better than a uniform random player ($p \ge 0.55$ vs $p \le 0.50$):
```bash
PYTHONPATH=. python -m backend.arena_sprt
```

---

## 🏆 Empirical Results & Search Compression

| Version | Architecture & Representations | Simulation CSE ($N_{50}$) | 95% Confidence Interval | Realized Wall-Clock Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **V4** | Dense MLP Heads (3-Channel) | 3,787.1 | Extrapolated | 0.55x |
| **V5 Clone** | FCN Head + Boundary Padding (5-Channel) | 25,615.7 | [366.2, 1,791,867.9] | 3.97x |
| **V5 Fine-Tune** | FCN Head + Boundary Padding (5-Channel) | **166,343.9** | **[89,923.4, 307,709.6]** | **21.55x** |

### Key Takeaways:
- **Search Density Scaling**: The V5 fine-tuned model compresses the strategic depth of over **166,000 Classic MCTS stochastic rollouts** into just **400 latent neural expansions** (a $+3.7818$ nat gain over V4, or a **43.9x search efficiency multiplier**).
- **Wall-Clock Acceleration**: Because neural tensor operations execute with high parallelism on GPU/SIMD hardware, 400 MuZero lookaheads execute in $\sim 0.18$s per move compared to $\sim 3.9$s for 166k rollouts on CPU, delivering a **21.55x real-world speedup**.

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
│   ├── hex_env.py                    # Gymnasium-style Hex environment & 5-channel observations
│   ├── muzero_nets.py                # FCN Representation, Dynamics & Prediction networks
│   ├── latent_mcts.py                # Latent space MCTS search engine & PUCT
│   ├── train.py                      # BPTT Replay Buffer & Self-Play CLI Trainer
│   ├── train_supervised.py           # Supervised Behavioral Cloning trainer
│   ├── generate_expert_data.py       # Classic MCTS trajectory generator
│   ├── generate_muzero_data.py       # MuZero self-play bootstrap dataset generator
│   ├── arena_logistic.py             # Active Sequential Experiment Design CSE benchmark
│   ├── arena_sprt.py                 # Wald's Sequential Probability Ratio Test
│   ├── verify_v5.py                  # Self-healing verification suite
│   ├── requirements.txt              # PyTorch, FastAPI, Uvicorn, WebSockets, plotext, scipy
│   └── runs/                         # Model checkpoint versions (v4_clone, v5_clone, v5_fine_tune)
├── paper/                            # LaTeX Documentation & Research Paper
│   ├── mcts_hex_paper.tex            # LaTeX paper source code
│   └── mcts_hex_paper.pdf            # Compiled PDF paper
├── hex-pwa/                          # Single Page Web Application
│   ├── index.html                    # SPA HTML entry point & UI layout
│   ├── style.css                     # Dark navy/cyan CSS design system
│   ├── app.js                        # Hex engine, MCTS search, WebSocket adapter
│   ├── sw.js                         # Service Worker with auto-update caching
│   └── manifest.json                 # Web App Manifest for PWA installation
├── pack_scripts.py                   # V5 Toolchain packer / unpacker
├── v5_toolchain.txt                  # Standalone bundled toolchain script archive
└── README.md
```

---

## 📜 License & Credits

- Created for exploring Monte Carlo Tree Search, Topology, and Deep Reinforcement Learning in Hex.
- Hosted on [GitHub Pages](https://tryggth.github.io/hex/).
