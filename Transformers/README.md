# CSI Prediction with Transformers

This repository contains the implementation of Transformer models for predicting Channel State Information (CSI) in wireless communication systems. Transformers, originally developed for natural language processing tasks, have been adapted to handle time series data and show promising results due to their ability to capture long-range dependencies.

## Introduction

Channel State Information (CSI) is vital for understanding the properties of a wireless communication channel. Accurate CSI prediction can significantly enhance the performance and reliability of wireless networks. The Transformer model, known for its self-attention mechanism, is particularly adept at handling sequences of data, making it an excellent candidate for CSI prediction.

## Transformer Model Overview

The Transformer model relies entirely on self-attention mechanisms to draw global dependencies between input and output, without using sequence-aligned RNNs or convolution. 

### Mathematical Representation

**Self-Attention:**
The self-attention mechanism in the Transformer computes the response at a position in a sequence by attending to all positions and taking their weighted average. Mathematically, it can be described as follows:

\[ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V \]

where \(Q\), \(K\), and \(V\) are the query, key, and value matrices derived from the input, and \(d_k\) is the dimension of the key vectors.

**Position-wise Feed-Forward Networks:**
Each layer of the Transformer contains a fully connected feed-forward network which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between:

\[ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 \]


## Contributors

- Rohit Viswam
- Rithvik Grandhi

