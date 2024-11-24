# Maximum Independent Set Oracle with Graph Convolutional Networks

## Overview

In this project, we aim to construct an **oracle** that predicts the probability that each node in a graph belongs to the Maximum Independent Set (MIS). These probabilities provide valuable insights into the structure of the graph and are particularly useful in applications where exact MIS solutions are computationally infeasible.

Our approach leverages **Graph Convolutional Networks (GCNs)** to encode graph structure and node features. By clustering nodes with similar feature representations and running a greedy MIS approximation within each cluster, we derive node-level probabilities as a proxy for the oracle's accuracy. This method addresses an open problem outlined in [this paper](https://arxiv.org/abs/placeholder). 

---

## Approach

### Graph Convolutional Networks (GCNs)

1. **Encoding Graph Structure**:  
   We use GCNs to learn embeddings for each node, capturing structural and feature information.

2. **Clustering Nodes**:  
   Nodes with similar embeddings are grouped into clusters. This ensures that nodes with similar roles in the graph are processed together.

3. **Greedy MIS Approximation**:  
   Within each cluster, a greedy algorithm is applied to approximate the MIS.

4. **Deriving Probabilities**:  
   The node-level probabilities of belonging to the MIS are computed based on the greedy results. These probabilities serve as a proxy for the oracle's accuracy.

---

## Features

- **Graph Representation Learning**: Encodes graph structure and node features using GCNs.
- **Clustering**: Groups nodes into clusters for localized MIS approximation.
- **Scalable Approximation**: Employs a greedy algorithm to efficiently estimate MIS probabilities.

---
