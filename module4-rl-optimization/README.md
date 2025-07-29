# Module 4: Reinforcement Learning for Crop ROI Optimization

## Objective & Research Gap
This module formulates agricultural planning as a multi-objective deep reinforcement learning (MO-DRL) problem. The goal is to train an agent that learns a dynamic policy to navigate the trade-offs between yield, profitability, and systemic risk.

## Methodology
1.  **MDP Formulation:** State (location, climate/market forecast), Action (crop choice).
2.  **Reward Function:** A weighted sum: $R = w_1 \cdot R_{yield} + w_2 \cdot R_{profit} - w_3 \cdot R_{risk}$.
3.  **Training:** A Proximal Policy Optimization (PPO) agent is trained in a custom Gymnasium environment on 20 years of historical Sri Lankan data.

## Evaluation Protocol
- **Quantitative:** Compare a Baseline Strategy vs. the RL Agent over a simulated 10-year period on Mean Annual Return, Return Volatility, and Incidence of Loss-Making Years.
- **Qualitative:** A learning curve plot (mean reward vs. training steps).

## Local Folder Structure
- `notebooks/`: For environment testing and agent hyperparameter tuning.
- `src/`: For the custom Gymnasium environment and reward function logic.
- `scripts/`: To run the full RL training loop.
- `data/`, `trained_models/`: Gitignored folders. Historical data and trained agent policies live on cloud storage.
