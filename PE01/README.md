This code implements a simple Q-Learning algorithm in Python using a reward matrix R and a Q-value matrix Q. The environment is defined as a 6-state space with specific allowed transitions and rewards. The agent learns optimal policies through exploration and exploitation by updating Q-values using the Bellman equation.

To enhance learning, the training loop is modified to begin only from states that have at least two possible actions, ensuring the agent is only trained in decision-relevant scenarios. This promotes more effective exploration and better policy convergence.

The training runs for 50,000 episodes, and the final Q-matrix is normalized and printed for analysis. Deprecation warnings are handled by ensuring scalar extraction from numpy arrays where necessary.