#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

class RBM:
    def __init__(self, n_visible, n_hidden, learning_rate=0.1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate

        # Initialize weights and biases
        self.weights = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.visible_bias = np.zeros(n_visible)
        self.hidden_bias = np.zeros(n_hidden)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sample_prob(self, probs):
        return (probs > np.random.random(probs.shape)).astype(float)

    def train(self, data, epochs=5000):
        for epoch in range(epochs):
            # Positive phase
            pos_hidden_probs = self.sigmoid(np.dot(data, self.weights) + self.hidden_bias)
            pos_hidden_states = self.sample_prob(pos_hidden_probs)
            pos_associations = np.dot(data.T, pos_hidden_probs)

            # Negative phase (reconstruction)
            neg_visible_probs = self.sigmoid(np.dot(pos_hidden_states, self.weights.T) + self.visible_bias)
            neg_hidden_probs = self.sigmoid(np.dot(neg_visible_probs, self.weights) + self.hidden_bias)

            neg_associations = np.dot(neg_visible_probs.T, neg_hidden_probs)

            # Update weights and biases
            self.weights += self.learning_rate * ((pos_associations - neg_associations) / data.shape[0])
            self.visible_bias += self.learning_rate * np.mean(data - neg_visible_probs, axis=0)
            self.hidden_bias += self.learning_rate * np.mean(pos_hidden_probs - neg_hidden_probs, axis=0)

        # Return reconstruction from the last epoch
        return neg_visible_probs

data = np.array([
    [1, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 1],
    [0, 1, 1, 0, 1, 0],
    [1, 1, 0, 1, 1, 0],
    [0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1]
])

# Create RBM model
rbm = RBM(n_visible=6, n_hidden=2, learning_rate=0.1)

# Train RBM and get reconstructed data
reconstructed_data = rbm.train(data, epochs=5000)

# Print reconstructed data (rounded to 0s and 1s)
print("Reconstructed data from RBM:")
print(np.round(reconstructed_data))

