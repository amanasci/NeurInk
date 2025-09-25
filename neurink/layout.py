"""
Advanced layout engine implementing the Sugiyama hierarchical graph drawing algorithm.

This module provides sophisticated layout algorithms for complex neural network
architectures, producing aesthetically pleasing and topologically correct diagrams.
"""

from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict, deque
import math
from .graph import Graph, build_graph_from_diagram


class DummyNode:
    """Represents a dummy node inserted for edge routing."""
    
    def __init__(self, name: str, original_edge: Tuple[str, str]):
        """
        Initialize a dummy node.
        
        Args:
            name: Unique name for the dummy node
            original_edge: The original edge (source, target) this dummy helps route
        """
        self.name = name
        self.original_edge = original_edge
        self.layer_type = "dummy"
        
    def get_shape_info(self) -> Dict[str, Any]:
        """Get shape info for rendering (dummy nodes are invisible)."""
        return {
            "type": "dummy",
            "display_text": "",
            "visible": False
        }


class AdvancedLayoutEngine:
    """
    Sophisticated multi-stage layout engine implementing the Sugiyama algorithm.
    
    The algorithm proceeds through 5 phases:
    1. Graph construction and cycle removal
    2. Node ranking (layer assignment) 
    3. Dummy node insertion for long edges
    4. Crossing reduction within ranks
    5. Coordinate assignment
    """
    
    def __init__(self):
        """Initialize the layout engine."""
        self.graph: Optional[Graph] = None
        self.ranks: Dict[str, int] = {}
        self.ordered_ranks: List[List[str]] = []
        self.dummy_nodes: List[DummyNode] = []
        self.coordinates: Dict[str, Tuple[float, float]] = {}
        
    def calculate_positions(self, diagram, styles: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate optimal positions for all nodes using the Sugiyama algorithm.
        
        Args:
            diagram: NeurInk Diagram object
            styles: Styling parameters for layout calculations
            
        Returns:
            Dictionary mapping node names to (x, y) coordinates
        """
        print("=== Phase 1: Graph Construction ===")
        self.graph = build_graph_from_diagram(diagram)
        print(f"Built graph with {len(self.graph.nodes)} nodes")
        print(self.graph)
        
        print("\n=== Phase 2: Node Ranking ===")
        self.ranks = self._calculate_longest_path_ranking()
        print(f"Assigned ranks to {len(self.ranks)} nodes")
        for rank in sorted(set(self.ranks.values())):
            nodes_in_rank = [node for node, r in self.ranks.items() if r == rank]
            print(f"Rank {rank}: {nodes_in_rank}")
        
        print("\n=== Phase 3: Dummy Node Insertion ===")
        self._insert_dummy_nodes()
        print(f"Inserted {len(self.dummy_nodes)} dummy nodes")
        
        print("\n=== Phase 4: Crossing Reduction ===")
        self._reduce_crossings()
        print(f"Optimized ordering for {len(self.ordered_ranks)} ranks")
        for i, rank_nodes in enumerate(self.ordered_ranks):
            print(f"Rank {i}: {rank_nodes}")
        
        print("\n=== Phase 5: Coordinate Assignment ===")
        self.coordinates = self._assign_coordinates(styles)
        print(f"Assigned coordinates to {len(self.coordinates)} nodes")
        
        return self.coordinates
    
    def _calculate_longest_path_ranking(self) -> Dict[str, int]:
        """
        Calculate node ranks using the longest path algorithm.
        
        This ensures minimum diagram height while respecting edge directions.
        
        Returns:
            Dictionary mapping node names to their rank (layer) numbers
        """
        if not self.graph:
            raise ValueError("Graph not initialized")
        
        # Calculate in-degrees for all nodes
        in_degrees = {node: self.graph.get_in_degree(node) for node in self.graph.nodes}
        
        # Initialize ranks and queue
        ranks = {node: 0 for node in self.graph.nodes}
        queue = deque([node for node, degree in in_degrees.items() if degree == 0])
        
        # Process nodes in topological order
        processed = 0
        while queue:
            node = queue.popleft()
            processed += 1
            
            # Update ranks of all successors
            for successor in self.graph.get_successors(node):
                ranks[successor] = max(ranks[successor], ranks[node] + 1)
                in_degrees[successor] -= 1
                
                # Add to queue when all predecessors are processed
                if in_degrees[successor] == 0:
                    queue.append(successor)
        
        if processed != len(self.graph.nodes):
            raise ValueError("Graph contains cycles - ranking failed")
        
        return ranks
    
    def _insert_dummy_nodes(self) -> None:
        """
        Insert dummy nodes to handle edges spanning multiple ranks.
        
        This breaks long edges into segments for proper routing.
        """
        self.dummy_nodes = []
        dummy_counter = 0
        
        # Create new graph with dummy nodes
        new_graph = Graph()
        
        # Add original nodes
        for node in self.graph.nodes:
            new_graph.add_node(node, self.graph.node_data.get(node))
        
        # Process each edge
        for source in self.graph.nodes:
            for target in self.graph.get_successors(source):
                source_rank = self.ranks[source]
                target_rank = self.ranks[target]
                
                if target_rank - source_rank > 1:
                    # This is a long edge - insert dummy nodes
                    prev_node = source
                    
                    # Create chain of dummy nodes
                    for rank in range(source_rank + 1, target_rank):
                        dummy_name = f"dummy_{source}_{target}_{dummy_counter}"
                        dummy_counter += 1
                        
                        # Create dummy node
                        dummy = DummyNode(dummy_name, (source, target))
                        self.dummy_nodes.append(dummy)
                        
                        # Add to graph and set rank
                        new_graph.add_node(dummy_name, dummy)
                        self.ranks[dummy_name] = rank
                        
                        # Connect previous node to dummy
                        new_graph.add_edge(prev_node, dummy_name)
                        prev_node = dummy_name
                    
                    # Connect last dummy to target
                    new_graph.add_edge(prev_node, target)
                else:
                    # Regular edge - add directly
                    new_graph.add_edge(source, target)
        
        # Replace graph with the new one containing dummy nodes
        self.graph = new_graph
    
    def _reduce_crossings(self, max_iterations: int = 20) -> None:
        """
        Reduce edge crossings using the barycenter heuristic.
        
        Args:
            max_iterations: Maximum number of optimization iterations
        """
        if not self.graph or not self.ranks:
            raise ValueError("Graph and ranks must be initialized")
        
        # Group nodes by rank
        max_rank = max(self.ranks.values())
        self.ordered_ranks = [[] for _ in range(max_rank + 1)]
        
        for node, rank in self.ranks.items():
            self.ordered_ranks[rank].append(node)
        
        # Initial ordering (preserve original order where possible)
        for rank_nodes in self.ordered_ranks:
            rank_nodes.sort()
        
        # Iterative improvement using barycenter heuristic
        for iteration in range(max_iterations):
            improved = False
            
            # Downward pass
            for rank in range(1, len(self.ordered_ranks)):
                new_ordering = self._calculate_barycenter_ordering(rank, direction="down")
                if new_ordering != self.ordered_ranks[rank]:
                    self.ordered_ranks[rank] = new_ordering
                    improved = True
            
            # Upward pass
            for rank in range(len(self.ordered_ranks) - 2, -1, -1):
                new_ordering = self._calculate_barycenter_ordering(rank, direction="up")
                if new_ordering != self.ordered_ranks[rank]:
                    self.ordered_ranks[rank] = new_ordering
                    improved = True
            
            # Stop if no improvement
            if not improved:
                print(f"Crossing reduction converged after {iteration + 1} iterations")
                break
    
    def _calculate_barycenter_ordering(self, rank: int, direction: str) -> List[str]:
        """
        Calculate optimal ordering for a rank using barycenter values.
        
        Args:
            rank: Rank number to optimize
            direction: "down" or "up" for sweep direction
            
        Returns:
            Optimally ordered list of nodes for the rank
        """
        nodes = self.ordered_ranks[rank]
        barycenters = []
        
        for node in nodes:
            if direction == "down" and rank > 0:
                # Calculate barycenter based on predecessors
                predecessors = self.graph.get_predecessors(node)
                if predecessors:
                    positions = []
                    for pred in predecessors:
                        if pred in self.ordered_ranks[rank - 1]:
                            positions.append(self.ordered_ranks[rank - 1].index(pred))
                    barycenter = sum(positions) / len(positions) if positions else 0
                else:
                    barycenter = 0
            elif direction == "up" and rank < len(self.ordered_ranks) - 1:
                # Calculate barycenter based on successors
                successors = self.graph.get_successors(node)
                if successors:
                    positions = []
                    for succ in successors:
                        if succ in self.ordered_ranks[rank + 1]:
                            positions.append(self.ordered_ranks[rank + 1].index(succ))
                    barycenter = sum(positions) / len(positions) if positions else 0
                else:
                    barycenter = 0
            else:
                barycenter = nodes.index(node)  # Maintain current position
            
            barycenters.append((barycenter, node))
        
        # Sort by barycenter values
        barycenters.sort(key=lambda x: x[0])
        return [node for _, node in barycenters]
    
    def _assign_coordinates(self, styles: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        """
        Assign final (x, y) coordinates to all nodes.
        
        Args:
            styles: Styling parameters for spacing and dimensions
            
        Returns:
            Dictionary mapping node names to (x, y) coordinates
        """
        coordinates = {}
        
        # Get layout parameters
        layer_width = styles.get("layer_width", 100)
        layer_height = styles.get("layer_height", 40)
        layer_spacing_x = styles.get("layer_spacing_x", 120)
        layer_spacing_y = styles.get("layer_spacing_y", 80)
        padding = styles.get("padding", 20)
        
        # Calculate Y coordinates (based on rank)
        for rank, nodes in enumerate(self.ordered_ranks):
            y = padding + rank * (layer_height + layer_spacing_y) + layer_height // 2
            
            # Calculate X coordinates (with centering)
            if not nodes:
                continue
                
            # Find the maximum width needed for centering
            max_width = max(len(rank_nodes) for rank_nodes in self.ordered_ranks)
            total_width = (max_width - 1) * layer_spacing_x + layer_width
            
            # Center this rank
            rank_width = (len(nodes) - 1) * layer_spacing_x + layer_width
            x_offset = (total_width - rank_width) / 2
            
            for i, node in enumerate(nodes):
                x = padding + x_offset + i * layer_spacing_x + layer_width // 2
                coordinates[node] = (x, y)
        
        return coordinates
    
    def get_edge_path(self, source: str, target: str) -> List[Tuple[float, float]]:
        """
        Get the path coordinates for an edge, including dummy nodes.
        
        Args:
            source: Source node name
            target: Target node name
            
        Returns:
            List of (x, y) coordinates defining the edge path
        """
        if source not in self.coordinates or target not in self.coordinates:
            return []
        
        # For direct connections (no dummy nodes)
        path = [self.coordinates[source], self.coordinates[target]]
        
        # TODO: Handle paths through dummy nodes for long edges
        # This would require tracking which dummy nodes belong to which original edge
        
        return path