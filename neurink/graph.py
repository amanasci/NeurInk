"""
Graph representation and utilities for advanced layout algorithms.

This module provides graph data structures and algorithms for implementing
sophisticated layout algorithms like the Sugiyama hierarchical graph drawing.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, deque


class Graph:
    """
    Directed graph representation for neural network architectures.
    
    Supports efficient operations for layout algorithms including:
    - Adjacency list representation
    - Cycle detection and removal
    - Topological sorting
    - Graph analysis utilities
    """
    
    def __init__(self):
        """Initialize an empty graph."""
        self.nodes: Set[str] = set()
        self.edges: Dict[str, List[str]] = defaultdict(list)  # successors
        self.reverse_edges: Dict[str, List[str]] = defaultdict(list)  # predecessors
        self.node_data: Dict[str, Any] = {}  # store layer objects and metadata
        
    def add_node(self, node_name: str, data: Any = None) -> None:
        """
        Add a node to the graph.
        
        Args:
            node_name: Unique identifier for the node
            data: Optional data associated with the node (e.g., Layer object)
        """
        self.nodes.add(node_name)
        if data is not None:
            self.node_data[node_name] = data
            
    def add_edge(self, source: str, target: str) -> None:
        """
        Add a directed edge from source to target.
        
        Args:
            source: Source node name
            target: Target node name
        """
        # Ensure nodes exist
        if source not in self.nodes:
            self.add_node(source)
        if target not in self.nodes:
            self.add_node(target)
            
        # Add edge to both representations
        if target not in self.edges[source]:
            self.edges[source].append(target)
        if source not in self.reverse_edges[target]:
            self.reverse_edges[target].append(source)
    
    def get_successors(self, node: str) -> List[str]:
        """Get list of successor nodes."""
        return self.edges[node].copy()
    
    def get_predecessors(self, node: str) -> List[str]:
        """Get list of predecessor nodes."""
        return self.reverse_edges[node].copy()
    
    def get_in_degree(self, node: str) -> int:
        """Get the in-degree of a node."""
        return len(self.reverse_edges[node])
    
    def get_out_degree(self, node: str) -> int:
        """Get the out-degree of a node."""
        return len(self.edges[node])
    
    def has_cycle(self) -> bool:
        """
        Detect if the graph has cycles using DFS.
        
        Returns:
            True if the graph contains cycles, False otherwise
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node: WHITE for node in self.nodes}
        
        def dfs_visit(node: str) -> bool:
            colors[node] = GRAY
            
            for successor in self.edges[node]:
                if colors[successor] == GRAY:  # Back edge found - cycle detected
                    return True
                elif colors[successor] == WHITE and dfs_visit(successor):
                    return True
                    
            colors[node] = BLACK
            return False
        
        for node in self.nodes:
            if colors[node] == WHITE:
                if dfs_visit(node):
                    return True
        
        return False
    
    def remove_cycles(self) -> List[Tuple[str, str]]:
        """
        Remove cycles by identifying and removing feedback edges.
        
        Returns:
            List of edges that were removed to break cycles
        """
        removed_edges = []
        
        while self.has_cycle():
            # Simple approach: find the first back edge and remove it
            WHITE, GRAY, BLACK = 0, 1, 2
            colors = {node: WHITE for node in self.nodes}
            found_cycle = False
            
            def find_back_edge(node: str) -> bool:
                nonlocal found_cycle
                if found_cycle:
                    return True
                    
                colors[node] = GRAY
                
                for successor in self.edges[node]:
                    if colors[successor] == GRAY:  # Back edge - remove it
                        self.edges[node].remove(successor)
                        self.reverse_edges[successor].remove(node)
                        removed_edges.append((node, successor))
                        found_cycle = True
                        return True
                    elif colors[successor] == WHITE and find_back_edge(successor):
                        return True
                        
                colors[node] = BLACK
                return False
            
            # Find and remove a back edge
            for node in self.nodes:
                if colors[node] == WHITE:
                    if find_back_edge(node):
                        break
        
        return removed_edges
    
    def topological_sort(self) -> List[str]:
        """
        Perform topological sorting using Kahn's algorithm.
        
        Returns:
            List of nodes in topologically sorted order
            
        Raises:
            ValueError: If the graph contains cycles
        """
        if self.has_cycle():
            raise ValueError("Cannot perform topological sort on cyclic graph")
        
        # Calculate in-degrees
        in_degrees = {node: self.get_in_degree(node) for node in self.nodes}
        
        # Initialize queue with nodes having zero in-degree
        queue = deque([node for node, degree in in_degrees.items() if degree == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            # Process all successors
            for successor in self.edges[node]:
                in_degrees[successor] -= 1
                if in_degrees[successor] == 0:
                    queue.append(successor)
        
        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycles - topological sort incomplete")
        
        return result
    
    def get_source_nodes(self) -> List[str]:
        """Get nodes with no incoming edges."""
        return [node for node in self.nodes if self.get_in_degree(node) == 0]
    
    def get_sink_nodes(self) -> List[str]:
        """Get nodes with no outgoing edges."""
        return [node for node in self.nodes if self.get_out_degree(node) == 0]
    
    def __str__(self) -> str:
        """String representation of the graph."""
        lines = [f"Graph with {len(self.nodes)} nodes and {sum(len(edges) for edges in self.edges.values())} edges:"]
        for node in sorted(self.nodes):
            successors = self.edges[node]
            if successors:
                lines.append(f"  {node} -> {successors}")
            else:
                lines.append(f"  {node} -> []")
        return "\n".join(lines)


def build_graph_from_diagram(diagram) -> Graph:
    """
    Convert a NeurInk Diagram object into a formal directed graph.
    
    Args:
        diagram: NeurInk Diagram object with layers and connections
        
    Returns:
        Graph representation of the neural network
        
    Raises:
        ValueError: If cycles are detected and cannot be resolved
    """
    graph = Graph()
    
    # Ensure all layers have unique names
    layer_names = set()
    for i, layer in enumerate(diagram.layers):
        if not hasattr(layer, 'name') or not layer.name or layer.name in layer_names:
            layer.name = f"{layer.layer_type}_{i}"
        layer_names.add(layer.name)
    
    # Add nodes (layers) to the graph
    for layer in diagram.layers:
        graph.add_node(layer.name, layer)
    
    # Add sequential edges (default flow)
    for i in range(len(diagram.layers) - 1):
        source = diagram.layers[i].name
        target = diagram.layers[i + 1].name
        graph.add_edge(source, target)
    
    # Add custom connections
    if hasattr(diagram, 'connections'):
        for connection in diagram.connections:
            source = connection.source_name
            target = connection.target_name
            
            # Verify nodes exist
            if source not in graph.nodes:
                print(f"Warning: Connection source '{source}' not found in graph")
                continue
            if target not in graph.nodes:
                print(f"Warning: Connection target '{target}' not found in graph")
                continue
                
            graph.add_edge(source, target)
    
    # Handle cycles
    if graph.has_cycle():
        print("Warning: Cycles detected in graph. Attempting to resolve...")
        removed_edges = graph.remove_cycles()
        print(f"Removed {len(removed_edges)} edges to resolve cycles: {removed_edges}")
        
        if graph.has_cycle():
            raise ValueError("Unable to resolve all cycles in the graph")
    
    return graph