# Softmax and One-Hot Encoding in Python

## Overview

This Python script demonstrates how to compute the **softmax** function on a vector of scores, verify if the resulting vector is a valid probability distribution, and convert the softmax output into a **one-hot encoded vector** by selecting the highest value.

The softmax function is commonly used in machine learning, especially in classification problems, to convert raw scores (logits) into probabilities that sum to 1.

## Features

- Compute softmax using NumPy for numerical stability.
- Validate the softmax output by checking:
  - All values are between 0 and 1 (inclusive).
  - The sum of all values is approximately 1 (±0.01 tolerance).
- Print the normalized softmax vector and its sum.
- Identify the highest softmax value and generate a one-hot encoded vector.
- Clear outputs showing each step.

## Code Structure

- `softmax(x)`: Computes softmax of input vector `x`.
- `is_softmax_valid(output)`: Validates if the softmax output is a proper probability distribution.
- Main code:
  - Defines input vector `y`.
  - Computes softmax and rounds values for display.
  - Prints normalized softmax and sum.
  - Checks validity of softmax output.
  - Creates a one-hot vector based on the max softmax value.
  - Prints the one-hot vector.

## How to Run

1. Make sure you have Python 3 installed.
2. Install NumPy if not already installed:
