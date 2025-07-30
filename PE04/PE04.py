#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# In[14]:


# Load dataset
df = pd.read_csv('V1.csv')


# In[15]:


# Features and labels
X = df.loc[:, 'broke':'shouted']
Y = df.loc[:, 'class']


# In[16]:


# Train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, Y)


# In[17]:


# Define test input
X_DL = [[9, 0, 9, 0]]


# In[18]:


# Predict class
prediction = knn.predict(X_DL)
print("The prediction is:", repr(prediction[0]))


# In[19]:


# Get distances and indices of the 3 nearest neighbors
distances, indices = knn.kneighbors(X_DL)


# In[20]:


print("\nNearest neighbors (distance, feature values, label):")
for dist, idx in zip(distances[0], indices[0]):
    features = X.iloc[idx].values
    label = Y.iloc[idx]
    print(f"Distance: {dist:.2f}, Features: {features}, Label: {label}")

