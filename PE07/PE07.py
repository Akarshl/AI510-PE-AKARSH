#!/usr/bin/env python
# coding: utf-8

# In[8]:


import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import random
import math
import matplotlib.pyplot as plt


# In[9]:


# Load and preprocess MNIST
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images  = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255
train_labels_cat = tf.keras.utils.to_categorical(train_labels, 10)
test_labels_cat  = tf.keras.utils.to_categorical(test_labels, 10)


# In[10]:


# Simple CNN model (unchanged structure)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels_cat, epochs=1, batch_size=64, validation_split=0.1)


# In[11]:


# Reward matrix and Q-table for RL (kept as-is)
R = np.array([[0, 1], [1, 0]])
Q = np.zeros_like(R, dtype=float)
gamma = 0.8


# In[12]:


# Logistic sigmoid function for CRLMM-inspired adjustment (kept for completeness)
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))


# In[13]:


# Predict function + CRLMM update, now returning reward so we can tally results
def predict_and_reward(image, true_label):
    # verbose=1 to show the "1/1 ... step" lines like in your expected output
    pred_probs = model.predict(image[np.newaxis, ...], verbose=1)[0]
    action = int(np.argmax(pred_probs))
    reward = 1 if action == int(true_label) else -1
    # Simple Q update (unchanged logic)
    Q[0, action % 2] += gamma * reward
    print(f"Predicted: {action}, True: {int(true_label)}, Reward: {reward}, Q-table: {Q}")
    return reward


# In[14]:


# ---- Simulation with tracking ----
num_tests = 100  # run at least 100 samples
correct = 0
incorrect = 0

for _ in range(num_tests):
    idx = random.randint(0, len(test_images) - 1)
    r = predict_and_reward(test_images[idx], test_labels[idx])
    if r == 1:
        correct += 1
    else:
        incorrect += 1

accuracy = correct / num_tests
print(f"\nCorrect: {correct}, Incorrect: {incorrect}, Accuracy: {accuracy:.2f}")

