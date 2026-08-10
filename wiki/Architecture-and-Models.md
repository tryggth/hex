# 🏛️ Architecture and Models

The Hex engine features a dual-mode system allowing seamless offline client-side evaluation alongside a high-performance deep reinforcement learning backend.

---

## 1. Offline Client-Side PWA
- **Language**: Vanilla JavaScript (zero external dependencies).
- **Win Detection**: $\mathcal{O}(\alpha(V))$ Disjoint-Set (Union-Find) data structure with path compression and union by rank.
- **MCTS Engine**: UCT tree search with random rollouts utilizing the **Nash Trick**.
- **Visualization**: HTML5 canvas rendering with fractional power-curve attention heatmaps ($t = (v / v_{\text{max}})^{0.5}$).

---

## 2. Hybrid PyTorch MuZero Engine
- **Framework**: PyTorch + FastAPI + WebSockets.
- **Networks**:
  - **Representation ($h_\theta$)**: Embeds raw Hex observation planes into continuous spatial latent states.
  - **Dynamics ($g_\theta$)**: Predicts next latent states and transition rewards given a latent state and candidate action.
  - **Prediction ($f_\theta$)**: Computes policy logits $\mathbf{p}$ and scalar position value $v \in [-1, 1]$.
- **Training Optimization**:
  - **BPTT Unrolled Trajectories**: Unrolls predictions $K=5$ steps into the future.
  - **Gradient Scaling**: $0.5\times$ gradient scaling attached to unrolled hidden state hooks to prevent exploding gradients.
  - **Minimax Value Inversion**: Strict adversarial zero-sum inversion (`value = -value`) ignoring intermediate rewards.
