# 🏆 Hex AI Competition

Welcome to the **Hex AI Tournament & Benchmarking Competition**!

This page will serve as the hub for tournament announcements, submission procedures, rules, and live leaderboard rankings.

---

## 📌 Overview

The competition pits custom reinforcement learning agents, neural networks, and search heuristics against each other on a standard **7x7 Hex board**. 

Agents are evaluated on:
1. **Head-to-Head Win Rate**: Fair pairwise match-ups swapping first-mover advantage (Red vs. Blue).
2. **Compute Simulation Equivalent (CSE / "IQ")**: Efficiency relative to a classical MCTS baseline.
3. **SPRT Reliability**: Statistical significance over uniform random baseline play.

---

## 📋 Competition Format & Rules (Draft)

- **Board Size**: 7x7 Rhombus Grid.
- **Player Colors**:
  - **Player 1 (Red)**: Connects Top-to-Bottom.
  - **Player 2 (Blue)**: Connects Left-to-Right.
- **Time Controls**: 1,500 ms max think time per move (evaluated over WebSocket or headless CLI).
- **Match Structure**: Equal number of games played as Red and Blue to eliminate first-mover advantage.

---

## 🚀 How to Prepare Your Agent

Participants can build agents using our open-source framework:
- **Baseline Classic MCTS**: Pure Monte Carlo rollouts backed by an $\mathcal{O}(\alpha(V))$ Union-Find engine (`backend/classic_mcts.py`).
- **MuZero Latent MCTS**: Learned spatial embeddings with BPTT unrolling (`backend/latent_mcts.py`).

Stay tuned for official registration instructions, submission deadlines, and submission protocol specifications!
