# Reinforcement Learning Sandbox: Cliff Walking

This project is a standalone, educational implementation of **Tabular Reinforcement Learning** algorithms in a classic gridworld environment. It demonstrates the fundamental differences between **On-Policy** (safe) and **Off-Policy** (risky/optimal) learning.

## Project Overview

**The Environment**: The "Cliff Walking" task (Sutton & Barto, Example 6.6).
- **Goal**: Navigate from Start (S) to Goal (G) as quickly as possible.
- **The Catch**: The optimal path runs directly along the edge of a "Cliff". Falling off incurs a massive penalty (-100) and resets the agent.
- **Why it matters**: This environment perfectly illustrates the risk-reward tradeoff in RL. An agent that strives for optimality (shortest path) risks catastrophic failure, while a safer agent settles for a longer path.

## Algorithms Implemented

All algorithms are implemented from scratch in `rl_hello_cliff_sarsa.py`.

1.  **SARSA (State-Action-Reward-State-Action)**  
    ***On-Policy***: Learns the value of the policy it *actually follows*, including its exploration mistakes.  
    * **Behavior**: Highly risk-averse. It learns that walking near the cliff is dangerous because epsilon-greedy exploration might cause it to fall.

2.  **Q-Learning**  
    ***Off-Policy***: Learns the value of the *optimal greedy policy*, assuming it will act perfectly in the future, regardless of its current exploration.  
    * **Behavior**: Optimistic and risky. It learns the shortest path hugging the cliff edge. In a deterministic world, this is optimal. In a stochastic one, it's dangerous.

3.  **Expected SARSA**  
    ***On-Policy (Expectation)***: Similar to SARSA but updates based on the *expected value* of the next action rather than a random sample.
    * **Behavior**: More stable than SARSA due to reduced variance in updates, but generally converges to a similar risk-averse policy in this setting.

## Environment Details

*   **Grid**: 4x12 grid.
*   **Start**: Bottom-left (3, 0).
*   **Goal**: Bottom-right (3, 11).
*   **Cliff**: The entire bottom row between Start and Goal (3, 1..10).
*   **Dynamics**:
    *   **Stochasticity**: 10% chance (`SLIP_PROB = 0.1`) that the agent slips and takes a random action instead of the intended one.
    *   **Rewards**: -1 per step, -100 for falling off the cliff.

## Learning Setup

*   **Episodes**: 500
*   **Learning Rate (Alpha)**: 0.5
*   **Discount Factor (Gamma)**: 1.0 (Undiscounted episodic task)
*   **Exploration (Epsilon)**:  
    *   **Decay Schedule**: Starts at 1.0 (100% random) and decays to 0.01 (1% random) over the training period.
    *   **Why Decay?**: Allows the agent to explore extensively early on to find the goal, then exploit its knowledge to refine the policy.

## Results Summary

When you run the script, you will observe distinct behaviors arising from the algorithm definitions:

| Algorithm | Learned Path | Risk Profile | Why? |
| :--- | :--- | :--- | :--- |
| **SARSA** | **Safe** (Row 0/1) | Low | It "fears" the cliff because its learning targets include the possibility of a slip or random exploration step. |
| **Q-Learning** | **Risky** (Row 2) | High | It assumes optimal future actions, ignoring the risk of its own exploration. It hugs the cliff, often falling due to the 10% slip chance. |
| **Expected SARSA**| **Safe** (Row 0/1) | Low | Similar to SARSA, it accounts for the policy's exploration probability, leading it to value safety over raw speed. |

## How to Run

No external dependencies required (standard Python library).

```bash
python3 rl_hello_cliff_sarsa.py
```

## What This Project Teaches

1.  **Tabular RL Basics**: Implementing value iteration with Q-tables.
2.  **On-Policy vs Off-Policy**: Does the agent learn about "what it does" (SARSA) or "what it *could* do" (Q-Learning)?
3.  **Risk Management**: How algorithmic choices fundamentally alter safety in dangerous environments.
4.  **Stochasticity**: How environment noise (probability of slipping) turns "optimal" paths into suboptimal or dangerous ones.

## Suggested Extensions

If you want to build on this:
*   **Double Q-Learning**: Address the maximization bias of Q-Learning.
*   **N-Step Methods**: Update based on N steps of experience instead of just one (TD(0)).
*   **Policy Iteration**: Solve the environment using Dynamic Programming (requires model knowledge).
