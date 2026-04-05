# Reinforcement Learning: Stochastic Control & MDP Solvers

This repository contains implementations of various Reinforcement Learning (RL) algorithms applied to classic Markov Decision Processes (MDPs). The projects demonstrate the transition from **Planning** (where environment dynamics are known) to **Learning** (where the agent learns from experience).

---

## 1. Stochastic Gridworld (Planning)
A $5 \times 5$ Gridworld implementation featuring teleportation dynamics, boundary penalties, and stochastic transitions.

### Applied Methods:

#### **A. Exact Analytical Solution (Matrix Inversion)**
The value function is solved directly by treating the Bellman Expectation Equation as a linear system. This provides the "ground truth" equilibrium for the 25-state space using direct linear algebra:
$$\mathbf{v} = (\mathbf{I} - \gamma\mathbf{P})^{-1} \mathbf{r}$$

#### **B. Iterative Policy Evaluation (Dynamic Programming)**
A bootstrapping approach that improves state-value estimates through successive synchronous sweeps of the state space.
* **Mechanism:** Synchronous DP updates based on the Bellman Expectation Equation.
* **Convergence:** Guaranteed fixed-point equilibrium with a tolerance of $\Delta < 10^{-6}$.

---

## 2. Blackjack: Optimal Control (Learning)
A simulation of the game of Blackjack where the environment dynamics (transition probabilities) are unknown to the agent.

### Applied Methods:

#### **Generalized Policy Iteration (GPI)**
The control problem is solved through an iterative process that alternates between evaluation and improvement:

* **First-Visit Monte Carlo Evaluation:** Estimates the action-value function $Q(s, a)$ by averaging returns following the **first visit** to a state-action pair per episode. This ensures unbiased estimates by avoiding intra-episode correlation.
* **Exploring Starts (MC-ES):** To ensure global convergence and comprehensive exploration of the state-action space, episodes are initialized with random starting states and actions.
* **Greedy Policy Improvement:** After each episode, the policy is updated to be greedy with respect to the current $Q$-estimates:
$$\pi(s) = \arg\max_{a} Q(s, a)$$

#### **Incremental Update Rule**
For computational efficiency, $Q$-values are updated using a running average, allowing the agent to learn incrementally from millions of episodes:
$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)} \left[ G - Q(S_t, A_t) \right]$$

---

## Technical Stack
* **Python 3.x**
* **NumPy** (Linear Algebra & Matrix Operations)
* **Matplotlib** (3D Value Function Visualization)
* **Gymnasium** (Environment Modeling)

## Key Results
* **Gridworld:** Visualized value propagation from high-reward teleporters across the grid.
* **Blackjack:** Derived the optimal decision boundary for "Hit vs. Stick" across 9.1 million training episodes, accounting for the **Usable Ace** logic.
