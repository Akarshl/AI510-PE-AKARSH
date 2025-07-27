#!/usr/bin/env python
# coding: utf-8

# In[10]:


import math
import numpy as np

# y is the vector of the scores of the lv vector in the warehouse example.
y = [0.0002, 0.2, 0.9, 0.0001, 0.4, 0.6]
print('0.Vector to be normalized',y)


# In[11]:


#Version 1 : Explicitly writing the softmax function for this case
y_exp = [math.exp(i) for i in y]
print("1", [i for i in y_exp])
print("2", [round(i, 2) for i in y_exp])
sum_exp_yi = sum(y_exp)
print("3", round(sum_exp_yi, 2))
print("4", [round(i) for i in y_exp])
softmax = [round(i / sum_exp_yi, 3) for i in y_exp]
print("5",softmax)


# In[12]:


#Version 2 : Explicitly but with no comments
y_exp = [math.exp(i) for i in y]
sum_exp_yi = sum(y_exp)
softmax = [round(i / sum_exp_yi, 3) for i in y_exp]
print("6, Normalized vector",softmax)


# In[13]:


# version 3
# Function to calculate softmax
def softmax(x):
    x = np.array(x)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Function to validate softmax output
def is_softmax_valid(output):
    within_range = all(0 <= i <= 1 for i in output)
    approx_sum = abs(sum(output) - 1.0) <= 0.01
    return within_range and approx_sum

# Input vector
y = [0.0002, 0.2, 0.9, 0.0001, 0.4, 0.6]

# Compute softmax
softmax_result = [round(float(val), 3) for val in softmax(y)]
rounded_softmax = [round(val, 3) for val in softmax_result]

# Print softmax results
print("Normalized softmax:", rounded_softmax)
print("Sum of softmax:", round(sum(softmax_result), 3))
print("Softmax is valid:", is_softmax_valid(softmax_result))

# One-hot encoding based on softmax
ohot = max(softmax_result)
ohotv = list(softmax_result)  # Create a mutable list copy

print("Highest value in the normalized vector:", round(ohot, 3))
print("One-hot vector based on highest softmax value:")

for i in range(len(ohotv)):
    ohotv[i] = 1 if ohotv[i] == ohot else 0

print("This is a vector that is an output of a one-hot function on a softmax vector:")
print(ohotv)

