import torch
import torch.nn.functional as F
import torch_geometric
from sklearn.cluster import KMeans
import pickle
from src.embedding_model import IterativeEmbeddingModel
from src.cluster_greedy_mis import greedy_mis_algorithm
from src.calculate_gurobi_mis import calculate_mis_with_gurobi
import networkx as nx

def prepare_edge_indices(graph):
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    anti_edges = nx.complement(graph).edges
    anti_edge_index = torch.tensor(list(anti_edges), dtype=torch.long).t().contiguous()
    return edge_index, anti_edge_index

def cluster_nodes_by_embeddings(embeddings, num_clusters=5):
    """
    Cluster nodes based on their embeddings using k-means clustering.

    Parameters:
    - embeddings (torch.Tensor): Tensor of node embeddings.
    - num_clusters (int): The number of clusters to form.

    Returns:
    - dict: A dictionary where keys are cluster indices and values are lists of node indices in that cluster.
    """
    # Convert embeddings to a numpy array for compatibility with sklearn
    embeddings_np = embeddings.detach().cpu().numpy()

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_np)

    # Group nodes by cluster labels
    clusters = {}
    for node_idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node_idx)
    
    return clusters

# Step 1: Load the trained model parameters
def load_trained_model(model_path, p):
    model = IterativeEmbeddingModel(p=p)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    return model

# Step 2: Load the graphs from the pickle file
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    graphs = []
    if isinstance(data, list):
        for graph_data in data:
            graph = nx.Graph()
            edge_index = graph_data['edge_index']
            edges = edge_index.t().tolist()
            graph.add_edges_from(edges)
            graphs.append(graph)
    else:
        graph = nx.Graph()
        edge_index = data['edge_index']
        edges = edge_index.t().tolist()
        graph.add_edges_from(edges)
        graphs.append(graph)
    return graphs

# Step 3: Test the model (get embeddings and perform clusterwise greedy MIS)
def test_model_on_graph(model, graph):
    # Prepare edge indices for neighbor and anti-neighbor aggregation
    edge_index, anti_edge_index = prepare_edge_indices(graph)
    
    # Initialize node embeddings
    initial_embeddings = torch.ones((len(graph.nodes), 3 * model.p), dtype=torch.float)
    
    # Generate embeddings using the trained model
    node_embeddings = model(initial_embeddings, edge_index, anti_edge_index)
    
    # Cluster nodes based on embeddings
    clusters = cluster_nodes_by_embeddings(node_embeddings)  # Implement clustering logic

    predicted_mis = {}
    nodes_in_mis = []
    nodes_unmarked = set(graph.nodes)
    # Perform greedy MIS within each cluster
    for cluster, nodes in clusters.items():
        valid_nodes = [nd for nd in nodes if nd in nodes_unmarked]
        subgraph = graph.subgraph(valid_nodes)
        
        # Run greedy MIS and update graph and predicted MIS
        updated_graph, predicted_subset = greedy_mis_algorithm(subgraph, graph, nodes_unmarked)
        
        # Update the original graph with the updated graph
        graph = updated_graph  # Ensure that the updated graph is reflected here
        
        # Update the predicted MIS
        for k, v in predicted_subset.items():
            predicted_mis[k] = v
            if v == 1:
                nodes_in_mis.append(k)
        
    return predicted_mis

# Step 4: Compare with Gurobi MIS solution
def compare_with_gurobi_mis(predicted_mis, graph):
    gurobi_mis = calculate_mis_with_gurobi(graph)
    
    # Calculate approximation ratio
    approximation_ratio = len(predicted_mis) / len(gurobi_mis)
    
    # Calculate accuracy (overlap of nodes in MIS)
    correct_predictions = len(set(predicted_mis) & set(gurobi_mis))
    accuracy = correct_predictions / len(graph.nodes)
    
    return approximation_ratio, accuracy

# Main testing pipeline for multiple graphs
def run_testing_pipeline(graph_file, model_file, p=8):
    print("Loading trained model...")
    model = load_trained_model(model_file, p)
    
    print("Loading graphs from pickle file...")
    graphs = load_data(graph_file)
    global_accuracy = 0
    count = 0
    for i, graph in enumerate(graphs):
        print(f"Testing model on Graph {i + 1}...")
        predicted_mis = test_model_on_graph(model, graph)
        
        print("Comparing with Gurobi MIS solution...")
        try:
            approximation_ratio, accuracy = compare_with_gurobi_mis(predicted_mis, graph)
            
            print(f"Graph {i + 1} Results:")
            print(f"Approximation Ratio: {approximation_ratio:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
            global_accuracy+=accuracy
            count+=1
        except:
            print("Model too large for size-limited license")
        print("Average accuracy = "+str(global_accuracy/count))
        print("-" * 40)

# Run testing
graph_file = '/Users/prishajain/Desktop/self_supervised_mis/dataset_buffer/collab_graphs.pickle'  # Replace with your graph file path
model_file = '/Users/prishajain/Desktop/self_supervised_mis/model_parameters/trained_model_parameters.pth'  # Path to your trained model

run_testing_pipeline(graph_file, model_file)
