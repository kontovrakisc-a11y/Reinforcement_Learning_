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

## 4. Racetrack: Off-Policy Monte Carlo Control (Learning)
A custom Racetrack solver that learns from experience using an off-policy Monte Carlo control algorithm. The notebook uses a Numba-optimized custom environment to handle velocity dynamics, stochastic action failure, wall collisions, and ray-cast finish detection.

### Applied Methods:
* **Off-policy Monte Carlo control** with weighted importance sampling.
* **Epsilon decay** and **action masking** to enforce valid motion and help long-trajectory learning.
* **Numba JIT acceleration** to train 500,000 episodes efficiently in seconds.
* **Custom reward shaping:** step penalty of $-1$ plus a large terminal reward on finish.

---

## 5. Windy Gridworld: SARSA Control (Learning)
A Windy Gridworld implementation from Sutton & Barto that learns the optimal path under column-dependent wind forces. The agent is trained on a $10\times 7$ grid using on-policy SARSA and an $\epsilon$-greedy policy.

### Applied Methods:
* **SARSA on-policy TD control** for direct policy learning.
* **Deterministic environment dynamics** augmented by column wind forces.
* **Epsilon-greedy exploration** with random tie-breaking to avoid directional bias.

---

## 6. Cliff Gridworld: TD Control Comparison (Learning)
A comparative study of TD control algorithms on the classic Cliff Walking task. The environment penalizes cliff falls heavily, and the notebook contrasts how different algorithms learn to trade off risk and path length.

### Applied Methods:
* **SARSA** for on-policy learning of the actual behavior policy.
* **Q-Learning** for off-policy estimation of the greedy optimal policy.
* **Expected SARSA** for lower-variance updates using expected next-state action values.
* **Cliff penalty analysis** to demonstrate safe versus risky policy behavior.

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
* **Racetrack:** Demonstrated Numba-accelerated off-policy Monte Carlo control, with action masking and decaying exploration to learn optimal racing trajectories from all start positions.
* **Windy Gridworld:** Learned a wind-aware optimal policy using SARSA and visualized the agent’s trajectory on a column-driven wind field.
* **Cliff Gridworld:** Compared SARSA, Q-Learning, and Expected SARSA, showing how on-policy learning produces safer trajectories while off-policy Q-learning can favor the cliff-edge optimal route.

