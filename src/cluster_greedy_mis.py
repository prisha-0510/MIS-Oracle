def greedy_mis_algorithm(subgraph, graph, nodes_unmarked):
    # Create a subgraph from the unmarked nodes to modify the original graph
    graph = graph.subgraph(nodes_unmarked).copy()  # Create a subgraph with only the unmarked nodes

    nodes = list(subgraph.nodes)
    mis = []
    
    while nodes:
        # Calculate the degree of the remaining nodes in the current subgraph (after previous removals)
        node_degrees = {node: len(list(graph.neighbors(node))) for node in nodes}
        
        # Get the node with the lowest degree
        lowest_degree_node = min(node_degrees, key=node_degrees.get)
        
        # Add the selected node to the MIS
        mis.append(lowest_degree_node)
        
        # Remove the selected node and its neighbors from the list of unmarked nodes
        nodes.remove(lowest_degree_node)
        nodes_unmarked.remove(lowest_degree_node)
        
        # Remove neighbors of the selected node from the unmarked nodes set
        neighbors = list(graph.neighbors(lowest_degree_node))
        for nd in neighbors:
            nodes_unmarked.discard(nd)  # Use discard to avoid errors if node is not in the set
        
        # Remove the selected node and its neighbors from the subgraph
        graph.remove_node(lowest_degree_node)
        for neighbor in neighbors:
            if neighbor in graph.nodes:
                graph.remove_node(neighbor)

        # Remove the selected node and its neighbors from the remaining nodes to be considered
        nodes = [node for node in nodes if node not in neighbors]
        
    # Return the updated graph and the MIS
    return graph, {node: 1 if node in mis else 0 for node in graph.nodes}
