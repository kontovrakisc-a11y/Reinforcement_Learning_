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
* **Convergence:** Guaranteed fixed-point equilibrium with a tolerance .


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


## 3. Jack's Car Rental: Optimal Control (Planning)
An implementation of the classic resource-allocation problem (Sutton & Barto, Example 4.2) where the environment dynamics are fully known. The agent manages a car rental company with two locations, deciding how many cars to transfer between them overnight to maximize expected future revenue.

### Applied Methods:

#### **Environment Modeling (Poisson Dynamics)**
The environment transitions are heavily stochastic, driven by Poisson-distributed daily requests and returns at each location. To ensure computational efficiency, the transition probabilities and expected immediate rewards are precomputed:
$$P(s' \mid s, a) = \sum \Pr(\text{rentals})\Pr(\text{returns}) \cdot \text{Indicator}(\text{valid transitions})$$

#### **Policy Iteration**
The optimal policy is found by alternating between two fundamental Dynamic Programming operations until the policy strictly converges (stabilizes):

* **Iterative Policy Evaluation:** Approximates the state-value function $V_\pi(s)$ for the current policy by sweeping through the state space and applying the Bellman Expectation Equation. The updates account for the immediate transfer costs and the expected subsequent rental rewards:
$$V_{k+1}(s) = \mathbb{E}\left[ R(s,a) - C(a) + \gamma V_k(s') \middle| s, a=\pi(s) \right]$$


* **Greedy Policy Improvement:** Once the value function is evaluated, the policy is updated via

$\pi_{\text{new}}(s)=\arg\max_{a\in\mathcal{A}(s)}\sum_{s'}P(s'|s,a)\left[R(s,a)+\gamma V(s')\right]$



---

## Technical Stack
* **Python 3.x**
* **NumPy** (Linear Algebra & Matrix Operations)
* **Matplotlib** (2D Contour Mapping & 3D Value Function Visualization  )
* **Gymnasium** (Environment Modeling)
* **SciPy** (`scipy.stats.poisson` for statistical distributions)

## Key Results
* **Gridworld:** Visualized value propagation from high-reward teleporters across the grid.
* **Blackjack:** Derived the optimal decision boundary for "Hit vs. Stick" across many training episodes, accounting for the **Usable Ace** logic.
* **Jack's Car Rental:** * Successfully reached optimal policy convergence in exactly 5 iterations.
    * Visualized the optimal action mapping (2D array/contour), demonstrating the non-linear decision boundaries for moving cars based on asymmetric request/return rates.
    * Plotted the 3D Optimal Value Function, illustrating the expected long-term value of varying starting inventories at both locations.

