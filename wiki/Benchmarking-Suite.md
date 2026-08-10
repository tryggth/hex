# 📊 Benchmarking Suite

The repository includes a suite of headless evaluation scripts to quantify neural network performance and simulation efficiency.

---

## 1. Statistical Hypothesis Testing (SPRT)
- **Script**: `backend/arena_sprt.py`
- **Method**: Wald's Sequential Probability Ratio Test.
- **Goal**: Evaluates whether MuZero is statistically significantly superior to a uniform random baseline (1-simulation Classic MCTS).
- **Log-Likelihood Ratio (LLR)**:
  $$\text{LLR} = \sum \ln \frac{P(\text{outcome} \mid H_1)}{P(\text{outcome} \mid H_0)}$$
- **Decision Bounds**: $A \approx 2.944$ (Accept $H_1$: MuZero > Random), $B \approx -2.944$ (Accept $H_0$: MuZero = Random).

---

## 2. Logistic CSE Parity Benchmark
- **Script**: `backend/arena_logistic.py`
- **Goal**: Measures the **Compute Simulation Equivalent (CSE)** (the exact Classic MCTS simulation count where win rate is 50%).
- **Model**: Fits a 4-parameter Sigmoid curve across 8 anchor points ($X \in [1, 5, 10, 25, 50, 100, 200, 400]$):
  $$P(\text{win}) = \frac{1}{1 + e^{-k(\ln(X) - \ln(\text{CSE}))}}$$
- **Terminal UI**: Features a split-screen dashboard with a live `plotext` regression chart on the left and an ANSI-colorized 7x7 Hex board on the right.
