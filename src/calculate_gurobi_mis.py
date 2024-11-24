import pickle
import gurobipy as gp
import networkx as nx

def calculate_mis_with_gurobi(graph, time_limit=3600):
    """
    Calculate the Maximum Independent Set (MIS) for a given graph using Gurobi solver.

    Parameters:
    - graph (networkx.Graph): The graph for which to compute the MIS.
    - time_limit (int): Time limit in seconds for Gurobi to find the solution.

    Returns:
    - dict: A dictionary where keys are node IDs, and values are binary values (1 for in MIS, 0 for not in MIS).
    """
    # Create a Gurobi model
    model = gp.Model("MIS")
    
    # Set parameters for the optimization model
    model.setParam("TimeLimit", time_limit)  # Set the time limit for optimization
    model.setParam("OutputFlag", 0)          # Disable output to console
    
    # Add variables for each node, 1 if in MIS, 0 if not
    node_vars = {node: model.addVar(vtype=gp.GRB.BINARY, name=f"x_{node}") for node in graph.nodes}
    
    # Objective: Maximize the sum of the node variables (maximize MIS size)
    model.setObjective(gp.quicksum(node_vars[node] for node in graph.nodes), gp.GRB.MAXIMIZE)
    
    # Add constraints: no two adjacent nodes can be in the MIS at the same time
    for u, v in graph.edges:
        model.addConstr(node_vars[u] + node_vars[v] <= 1, f"edge_{u}_{v}")
    
    # Optimize the model
    model.optimize()
    
    # Return the solution as a dictionary with 1 if the node is in MIS, 0 otherwise
    return {node: int(node_vars[node].X > 0.5) for node in graph.nodes}

def calculate_and_save_gurobi_mis(file_path, output_path, time_limit=3600):
    """
    Calculate the MIS for multiple graphs and save the results.

    Parameters:
    - file_path (str): Path to the pickle file containing graph data.
    - output_path (str): Path to save the output pickle file with MIS results.
    - time_limit (int): Time limit for Gurobi to find the MIS for each graph.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)  # Load the dataset
    
    gurobi_mis_solutions = []  # List to store MIS solutions
    
    for graph_data in data:
        # Create the graph from the edge list
        graph = nx.Graph()
        edge_index = graph_data['edge_index']
        
        # If the edge_index is in tensor format, convert it to list of edges
        edges = edge_index.t().tolist()
        graph.add_edges_from(edges)
        
        # Calculate the MIS using Gurobi
        mis_solution = calculate_mis_with_gurobi(graph, time_limit)
        gurobi_mis_solutions.append(mis_solution)
    
    # Save the MIS solutions to a pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(gurobi_mis_solutions, f)

    print(f"Calculated and saved Gurobi MIS solutions to {output_path}")
