# src/embedding_model.py
import torch
import torch.nn as nn

class IterativeEmbeddingModel(nn.Module):
    def __init__(self, p):
        super(IterativeEmbeddingModel, self).__init__()
        self.p = p
        self.theta1 = nn.Parameter(torch.randn(3 * p, p))
        self.theta2 = nn.Parameter(torch.randn(3 * p, p))
        self.theta3 = nn.Parameter(torch.randn(3 * p, p))

    def forward(self, node_embeddings, edge_index, anti_edge_index, num_iterations=2):
        for _ in range(num_iterations):
            current_embedding = node_embeddings
            neighbors_embedding_sum = self.aggregate_embeddings(current_embedding, edge_index)
            anti_neighbors_embedding_sum = self.aggregate_embeddings(current_embedding, anti_edge_index)
            new_embedding = torch.cat([
                torch.matmul(current_embedding, self.theta1),
                torch.matmul(neighbors_embedding_sum, self.theta2),
                torch.matmul(anti_neighbors_embedding_sum, self.theta3)
            ], dim=1)
            node_embeddings = new_embedding
        return node_embeddings

    def aggregate_embeddings(self, embeddings, edge_index):
        row, col = edge_index
        return torch.zeros_like(embeddings).index_add_(0, row, embeddings[col])
