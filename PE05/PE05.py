#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


# Load data
df = pd.read_csv('data_BC.csv')
print("Blocks of the Blockchain")
print(df.head())


# In[3]:


# Define features and label
features = ['DAY', 'STOCK', 'BLOCKS']
label = 'DEMAND'


# In[4]:


# Function to evaluate model
def evaluate_model(features_subset):
    X = df[features_subset]
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


# In[5]:


# Evaluate with all features
acc_all = evaluate_model(['DAY', 'STOCK', 'BLOCKS'])
print(f"Model with all features: {round(acc_all, 2)}")

# Evaluate without each feature
acc_no_day = evaluate_model(['STOCK', 'BLOCKS'])
print(f"Model without 'DAY': {round(acc_no_day, 2)}")

acc_no_stock = evaluate_model(['DAY', 'BLOCKS'])
print(f"Model without 'STOCK': {round(acc_no_stock, 2)}")

acc_no_blocks = evaluate_model(['DAY', 'STOCK'])
print(f"Model without 'BLOCKS': {round(acc_no_blocks, 2)}")

