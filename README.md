# Requirement Clustering and Visualization Script

## Overview
This script processes a list of business requirements, groups similar ones into clusters, and visualizes the results. It uses OpenAI for embeddings, Pinecone for storing and querying vectors, and K-Means for clustering.

---

## Features
1. **Embedding Generation**: Converts requirements into vector embeddings using OpenAI.
2. **Clustering**: Groups similar requirements using K-Means.
3. **Visualization**:
   - A scatter plot (`clusters.png`) shows relationships in 2D space.
   - A JSON file (`clusters.json`) groups requirements by cluster.

---

## How It Works
1. **Generate or Load Embeddings**:
   - Requirements are sent to OpenAI for embedding.
   - Embeddings are stored or loaded from Pinecone.

2. **Cluster Requirements**:
   - Embeddings are retrieved and grouped into clusters using K-Means.

3. **Visualize Results**:
   - Outputs:
     - `clusters.png`: A scatter plot.
     - `clusters.json`: A JSON file with grouped requirements.
---

