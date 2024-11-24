import copy
def greedy_mis_algorithm(graph):

    nodes = list(graph.nodes)
    g_copy = copy.deepcopy(graph)
    mis = []
    while nodes:
        # Calculate the degree of the remaining nodes in the current subgraph (after previous removals)
        node_degrees = {node: len(list(g_copy.neighbors(node))) for node in nodes}
        
        # Get the node with the lowest degree
        lowest_degree_node = min(node_degrees, key=node_degrees.get)
        
        # Add the selected node to the MIS
        mis.append(lowest_degree_node)
        
        # Remove the selected node and its neighbors from the list of unmarked nodes
        nodes.remove(lowest_degree_node)
        
        # Remove neighbors of the selected node from the unmarked nodes set
        neighbors = list(g_copy.neighbors(lowest_degree_node))
        # Remove the selected node and its neighbors from the subgraph
        g_copy.remove_node(lowest_degree_node)
        for neighbor in neighbors:
            if neighbor in g_copy.nodes:
                g_copy.remove_node(neighbor)

        # Remove the selected node and its neighbors from the remaining nodes to be considered
        nodes = [node for node in nodes if node not in neighbors]
        
    # Return the updated graph and the MIS
    return {node: 1 if node in mis else 0 for node in graph.nodes}
