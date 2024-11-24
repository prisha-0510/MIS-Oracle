# src/train_model.py
import torch.optim as optim
import torch
import torch.nn as nn
import networkx as nx
from src.classical_greedy import greedy_mis_algorithm
from src.embedding_model import IterativeEmbeddingModel

def prepare_edge_indices(graph):
    # Ensure nodes are indexed consecutively from 0
    graph = nx.convert_node_labels_to_integers(graph)
    
    # Prepare edge_index
    edges = list(graph.edges)
    if len(edges) == 0:
        print("case 1 ")
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        print("case 2 ")
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Prepare anti_edge_index
    anti_edges = list(nx.complement(graph).edges)
    if len(anti_edges) == 0:
        anti_edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        anti_edge_index = torch.tensor(anti_edges, dtype=torch.long).t().contiguous()
    
    return edge_index, anti_edge_index


def train_model_across_graphs(model, graphs, num_epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        total_loss = 0
        for graph in graphs:
            optimizer.zero_grad()
            initial_embeddings = torch.ones((len(graph.nodes), 3 * model.p), dtype=torch.float)
            edge_index, anti_edge_index = prepare_edge_indices(graph)
            mis_labels = greedy_mis_algorithm(graph)
            labels = torch.tensor([mis_labels[node] for node in graph.nodes], dtype=torch.float)
            refined_embeddings = model(initial_embeddings, edge_index, anti_edge_index)
            predictions = refined_embeddings.mean(dim=1)
            # Apply sigmoid to convert logits to probabilities between 0 and 1
            predictions = torch.sigmoid(predictions)
            # print(predictions)
            # print(labels)
            loss = nn.BCELoss()(predictions, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            # break
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(graphs):.4f}')
