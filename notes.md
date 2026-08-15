# Topics to Learn

## Graph Representation
* Nodes, edges
* Adjacency matrix
* Edge list
* Degree matrix
* Sparse vs dense graphs
* Directed/undirected graphs
* Bipartite graphs

## Graph Features & Embeddings
* Node features
* Edge features
* Graph-level features
* Initial node embeddings
* Why embeddings are needed

## Neighborhood & Aggregation
* 1-hop, 2-hop, K-hop neighbors
* Mean/sum/max aggregation
* Self-loops
* Neighborhood sampling

## Message Passing Neural Networks (MPNN)
* Message function
* Aggregation function
* Update function
* Multiple message-passing layers
* Over-smoothing

## GCN Mathematics
* Graph convolution
* Normalized adjacency matrix
* Why D^−1/2 AD^−1/2 appears
* Step-by-step numerical example

## GNN Architecture Concepts
* Input → message passing → hidden representation → output
* Number of GNN layers
* Receptive field
* Node classification
* Link prediction
* Graph classification

## GNN Training
* Forward propagation on graphs
* Loss calculation
* Backpropagation through message passing
* Gradient flow
* Full-batch vs mini-batch training

## PyTorch Geometric Basics
* Data
* edge_index
* x
* batch
* MessagePassing
* GCNConv
* Building a tiny GNN

## Graph Recommendation Fundamentals
* User-item graph
* Bipartite graph
* Collaborative filtering
* Implicit feedback
* Link prediction
* User/item embeddings

## LightGCN
* Why LightGCN exists
* LightGCN propagation
* Layer-wise embeddings
* Final embeddings
* User-item scoring
* BPR loss
* Negative sampling