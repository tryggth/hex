# ⬡ Hex MCTS — AI Board Game & PWA Demo

[![PLAY NOW — Live PWA Demo](https://img.shields.io/badge/PLAY_NOW-Live_PWA_Demo-00FFFF?style=for-the-badge&logo=google-chrome&logoColor=0a0e1a)](https://tryggth.github.io/hex/)
[![DOCUMENTATION — PDF Paper](https://img.shields.io/badge/DOCUMENTATION-PDF_Paper-FF4444?style=for-the-badge&logo=adobe-acrobat-reader&logoColor=white)](paper/mcts_hex_paper.pdf)
[![Deploy to GitHub Pages](https://github.com/tryggth/hex/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/tryggth/hex/actions/workflows/deploy-pages.yml)
[![PWA Self-Updating](https://img.shields.io/badge/PWA-Self--Updating-00FFFF?style=flat-square)](https://tryggth.github.io/hex/)

An installable, self-updating **Progressive Web App (PWA)** demonstrating real-time **Monte Carlo Tree Search (MCTS)** for the classic two-player strategy game **Hex**.

🎮 **Live Web Application**: [https://tryggth.github.io/hex/](https://tryggth.github.io/hex/)  
📄 **Mathematical Paper (PDF)**: [**`paper/mcts_hex_paper.pdf`**](paper/mcts_hex_paper.pdf) — *"Monte Carlo Tree Search and Hex: A Topological and Probabilistic Exploration"*

---

## 🌟 Overview & Features

Hex is a connection game played on a rhombus grid of hexagonal cells. The **Human (Red)** attempts to form an unbroken chain of connected stones from **Top to Bottom**, while the **AI (Blue)** attempts to connect **Left to Right**. Draws are mathematically impossible.

This app demonstrates how a pure **Monte Carlo Tree Search (MCTS)** algorithm discovers winning strategy **from scratch in real-time**—without any pre-computed opening books, lookup tables, or neural networks.

### Key Features

* **📄 Academic & Mathematical Documentation**:
  - Full research paper included in [`paper/mcts_hex_paper.tex`](paper/mcts_hex_paper.tex) and compiled as [`paper/mcts_hex_paper.pdf`](paper/mcts_hex_paper.pdf).
  - Covers topological equivalence to the Brouwer Fixed-Point Theorem, Nash's Strategy-Stealing argument, UCB1/UCT bandit formulation, log-factorial path computation, and non-linear heatmap scaling.
* **🎨 Dynamic Heatmap Visualization**:
  - The interior background fill of candidate hexes dynamically scales from **soft white (`#FFFFFF`)** for initial exploration up to **electric cyan/blue (`#0088FF`)** for top candidate moves ($v / v_{\max}$).
  - Number text colors remain constant dark navy (`#050a14`) for maximum legibility.
* **⏸️ Pause & Inspect AI**:
  - Halts search iterations and freezes the heatmap visit counts on the board.
  - Reveals **The Combinatorial Explosion** cosmic panel, calculating remaining path factorials ($E!$) via logarithmic sum ($\sum \log_{10} i$) and comparing the scale against cosmic benchmarks (e.g. atoms in the universe, Shannon Number).
* **⏹️ "Stop Strategizing" (Anytime Algorithm Demonstration)**:
  - Demonstrates that MCTS is an *Anytime Algorithm*.
  - Immediately interrupts search iterations (even when paused) and executes the best move found up to that instant.
* **⚙️ Dynamic Real-Time Controls**:
  - **Board Size**: $5\times5$ to $11\times11$ (Default: **`7×7`**).
  - **Thinking Time**: $500\text{ms}$ to $8000\text{ms}$ (Default: **`1500ms`**).
  - **Exploration ($c$)**: $0.0$ to $5.0$ (Default: **`1.40`**).
* **📱 Installable & Self-Updating PWA**:
  - Can be installed natively on desktop or mobile devices.
  - CI/CD workflow automatically bumps version numbers (`v2.1.<build>`) and Service Worker cache (`v<build>`) on every push to `main` for instant client updates.
* **📖 Interactive Educational Modal**:
  - Built-in "How It Works" documentation including math references to [Wolfram MathWorld: Game of Hex](https://mathworld.wolfram.com/GameofHex.html).

---

## 📁 Repository Structure

```
hex/
├── paper/                            # LaTeX Documentation & Paper
│   ├── mcts_hex_paper.tex            # LaTeX paper source
│   └── mcts_hex_paper.pdf            # Compiled PDF paper
├── hex-pwa/                          # Single Page Web Application
│   ├── index.html                    # SPA HTML entry point & UI layout
│   ├── style.css                     # Dark navy/cyan CSS design system
│   ├── app.js                        # Hex engine, MCTS search, Nash Trick rollouts, log10 math
│   ├── sw.js                         # Versioned Service Worker with auto-update caching
│   ├── manifest.json                 # Web App Manifest for standalone installation
│   ├── generate_icon.py              # Icon generator script
│   └── icons/                        # PWA app icons (192×192 and 512×512 PNGs)
├── .github/workflows/
│   ├── deploy-pages.yml              # CI/CD GitHub Actions Pages deployment & auto-versioning
│   └── compile-paper.yml             # CI/CD GitHub Actions LaTeX PDF compilation workflow
├── generate_hex_bg.py                # Presentation slide background generator (1920×1080)
├── generate_explosion_v2.py          # Minimax vs MCTS decision tree graphic generator (3840×2160)
└── README.md
```

---

## 💻 Local Setup & Running

No build step or external dependencies are required to run the web application locally.

1. Clone the repository:
   ```bash
   git clone https://github.com/tryggth/hex.git
   cd hex
   ```

2. Start a local static HTTP server:
   ```bash
   python3 -m http.server 8000 --directory hex-pwa
   ```

3. Open your browser to `http://localhost:8000`.

---

## 🔬 Technical Implementation Details

* **Win Condition Detection**: Uses a **Union-Find (Disjoint Set)** data structure with path compression and rank optimization over $N$ cells plus 4 virtual border sentinel nodes ($O(1)$ amortized per connection).
* **Simulation Rollouts**: Uses the **Nash Trick** (randomly shuffling remaining empty cells and evaluating full-board state) for $O(N)$ fast rollouts.
* **Non-Blocking Execution**: MCTS loop is chunked into 80-iteration batches yielding to the main thread via `setTimeout(0)`, ensuring a smooth 60 FPS UI.
* **Logarithmic Factorial Calculation**: Calculates $E!$ via $\sum_{i=1}^E \log_{10}(i)$ to prevent double-precision floating-point overflow for large boards up to $11\times11$.

---

## 📊 Presentation Graphics

This repository also includes Python scripts for generating presentation slide graphics:

* `python3 generate_hex_bg.py`: Outputs `hex_bg.png` (abstract fading hexagonal grid).
* `python3 generate_explosion_v2.py`: Outputs `combinatorial_explosion_v2.png` (Minimax vs MCTS decision tree comparison).

---

## 📜 License & Credits

- Created for demonstrating Monte Carlo Tree Search concepts in Hex.
- Hosted on [GitHub Pages](https://tryggth.github.io/hex/).
