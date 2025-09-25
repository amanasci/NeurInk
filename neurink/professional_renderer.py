"""
Professional SVG renderer for neural network diagrams.

This module provides a sophisticated rendering engine that creates
publication-quality diagrams similar to NN-SVG with proper skip connections,
professional styling, and advanced layout algorithms.
"""

from typing import List, Dict, Any, Tuple, Optional, Set
from .layer import Layer
from .themes import Theme
from .layout import AdvancedLayoutEngine


class ProfessionalSVGRenderer:
    """
    Professional SVG renderer with advanced layout and styling capabilities.
    
    Creates publication-quality neural network diagrams with:
    - Proper skip connection visualization
    - Advanced 3D styling and effects
    - Smart layout algorithms for complex architectures
    - Professional typography and spacing
    """
    
    def __init__(self):
        """Initialize the professional renderer."""
        self.layout = "horizontal"
        self.layout_engine = AdvancedLayoutEngine()
        
    def render_diagram(self, diagram, theme: Theme) -> str:
        """
        Render a complete diagram with advanced features to professional-quality SVG.
        
        Args:
            diagram: Diagram object with layers, connections, and groups
            theme: Theme object for styling
            
        Returns:
            Professional SVG content as string
        """
        if not diagram.layers:
            return self._empty_svg(theme)
            
        colors = theme.get_colors()
        styles = theme.get_styles()
        typography = theme.get_typography()
        
        # Calculate advanced layout using Sugiyama algorithm
        positions = self.layout_engine.calculate_positions(diagram, styles)
        
        # Filter out dummy nodes for rendering (they're invisible)
        visible_positions = {name: pos for name, pos in positions.items() 
                           if not name.startswith('dummy_')}
        
        # Calculate canvas size with proper margins for groups
        canvas_width, canvas_height = self._calculate_canvas_size_with_groups(
            visible_positions, diagram.groups, styles)
        
        # Generate professional SVG with advanced styling
        svg_parts = []
        svg_parts.append(self._professional_svg_header(canvas_width, canvas_height, colors))
        svg_parts.append(self._advanced_svg_defs(theme))
        
        # Render layer groups first (as background)
        if diagram.groups:
            svg_parts.append(self._render_layer_groups(diagram.groups, diagram.layers, visible_positions, colors, styles))
        
        # Render advanced connections using layout engine
        if len(diagram.layers) > 1 or diagram.connections:
            svg_parts.append(self._render_advanced_connections(
                diagram, self.layout_engine, positions, colors, styles))
        
        # Render layers with advanced 3D styling (skip dummy nodes)
        for layer in diagram.layers:
            if layer.name in visible_positions:
                x, y = visible_positions[layer.name]
                svg_parts.append(self._render_professional_layer(
                    layer, x, y, colors, styles, typography))
            
        svg_parts.append("</svg>")
        
        return "\n".join(svg_parts)

    def render(self, layers: List[Layer], theme: Theme) -> str:
        """
        Render layers to professional-quality SVG (backward compatibility).
        
        Args:
            layers: List of layer objects to render
            theme: Theme object for styling
            
        Returns:
            Professional SVG content as string
        """
        if not layers:
            return self._empty_svg(theme)
            
        colors = theme.get_colors()
        styles = theme.get_styles()
        typography = theme.get_typography()
        
        # Analyze architecture for skip connections
        architecture_graph = self._analyze_architecture(layers)
        
        # Calculate professional layout with proper skip connections
        positions = self._calculate_professional_layout(layers, architecture_graph, styles)
        
        # Calculate canvas size with proper margins
        canvas_width, canvas_height = self._calculate_canvas_size(positions, styles)
        
        # Generate professional SVG with advanced styling
        svg_parts = []
        svg_parts.append(self._professional_svg_header(canvas_width, canvas_height, colors))
        svg_parts.append(self._advanced_svg_defs(theme))
        
        # Render professional connections with proper skip connection paths
        if len(layers) > 1:
            svg_parts.append(self._render_professional_connections(
                layers, architecture_graph, positions, colors, styles))
        
        # Render layers with advanced 3D styling
        for i, layer in enumerate(layers):
            x, y = positions[i]
            svg_parts.append(self._render_professional_layer(
                layer, x, y, colors, styles, typography))
            
        svg_parts.append("</svg>")
        
        return "\n".join(svg_parts)
    
    def _analyze_architecture(self, layers: List[Layer]) -> Dict[str, Any]:
        """
        Analyze the architecture to identify skip connections and branching patterns.
        
        Returns:
            Architecture graph with skip connection information
        """
        graph = {
            'branches': {},  # branch_name -> start_index
            'merges': {},    # merge_index -> branch_name
            'skip_paths': [], # List of (start_idx, end_idx, branch_name)
            'main_path': [],  # List of layer indices on main path
        }
        
        for i, layer in enumerate(layers):
            if layer.layer_type == "branch":
                graph['branches'][layer.branch_name] = i
            elif layer.layer_type == "merge" and hasattr(layer, 'merge_with') and layer.merge_with:
                graph['merges'][i] = layer.merge_with
                if layer.merge_with in graph['branches']:
                    start_idx = graph['branches'][layer.merge_with]
                    graph['skip_paths'].append((start_idx, i, layer.merge_with))
        
        # Identify main path (layers not involved in skip connections)
        skip_layer_indices = set()
        for start_idx, end_idx, _ in graph['skip_paths']:
            for idx in range(start_idx, end_idx + 1):
                skip_layer_indices.add(idx)
        
        graph['main_path'] = [i for i in range(len(layers)) if i not in skip_layer_indices]
        
        return graph
        
    def _calculate_professional_layout(self, layers: List[Layer], 
                                     architecture_graph: Dict[str, Any], 
                                     styles: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        Calculate sophisticated layout for professional visualization.
        
        This creates proper Y-offsets for skip connections and ensures
        that the diagram clearly shows the architectural structure.
        """
        positions = []
        
        # Base positioning
        x_start = styles["padding"] + styles["layer_width"] // 2
        y_base = styles["padding"] + styles["layer_height"] // 2
        
        # Track current Y position for branching
        current_y_offset = 0
        skip_y_offsets = {}  # branch_name -> y_offset
        
        for i, layer in enumerate(layers):
            x = x_start + i * styles["layer_spacing_x"]
            
            if layer.layer_type == "branch":
                # Create branch with Y offset for skip connection visualization
                branch_offset = styles["layer_spacing_y"]
                skip_y_offsets[layer.branch_name] = branch_offset
                y = y_base + branch_offset
                current_y_offset = branch_offset
            elif layer.layer_type == "merge" and hasattr(layer, 'merge_with'):
                # Merge back to main path or branch path
                if layer.merge_with and layer.merge_with in skip_y_offsets:
                    y = y_base + skip_y_offsets[layer.merge_with]
                else:
                    y = y_base
                current_y_offset = 0
            else:
                # Regular layer follows current offset
                y = y_base + current_y_offset
                
            positions.append((x, y))
            
        return positions
        
    def _professional_svg_header(self, width: int, height: int, colors: Dict[str, str]) -> str:
        """Generate professional SVG header with enhanced styling."""
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"
     style="font-family: 'Source Sans Pro', Arial, sans-serif; background: {colors['background']};">
  <rect width="100%" height="100%" fill="{colors['background']}"/>'''
  
    def _advanced_svg_defs(self, theme: Theme) -> str:
        """Generate advanced SVG definitions with professional styling."""
        colors = theme.get_colors()
        styles = theme.get_styles()
        
        # Check if this is the NNSVG theme
        is_nnsvg = 'shadow' in colors
        
        defs = f'''  <defs>
    <!-- Professional arrowheads -->
    <marker id="arrowhead" markerWidth="12" markerHeight="8" 
            refX="12" refY="4" orient="auto" markerUnits="strokeWidth">
      <polygon points="0 0, 12 4, 0 8" 
               fill="{colors['connection']}" 
               stroke="none"
               style="filter: drop-shadow(1px 1px 1px rgba(0,0,0,0.2))"/>
    </marker>
    
    <!-- Skip connection arrowhead -->
    <marker id="skip-arrowhead" markerWidth="12" markerHeight="8" 
            refX="12" refY="4" orient="auto" markerUnits="strokeWidth">
      <polygon points="0 0, 12 4, 0 8" 
               fill="{colors['connection']}" 
               stroke="none" opacity="0.7"
               style="filter: drop-shadow(1px 1px 1px rgba(0,0,0,0.2))"/>
    </marker>'''
    
        if is_nnsvg:
            # Advanced filters and effects
            defs += f'''
    
    <!-- Advanced shadow filter -->
    <filter id="layer-shadow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="3"/>
      <feOffset dx="2" dy="4" result="offset"/>
      <feFlood flood-color="#000000" flood-opacity="0.25"/>
      <feComposite in2="offset" operator="in"/>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    
    <!-- Professional layer styling -->
    <filter id="layer-effects" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="2"/>
      <feOffset dx="1" dy="2" result="offset"/>
      <feFlood flood-color="#000000" flood-opacity="0.15"/>
      <feComposite in2="offset" operator="in"/>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>'''
            
            # Professional gradients for each layer type
            layer_types = ['input', 'conv', 'dense', 'output', 'flatten', 'dropout', 
                          'attention', 'layernorm', 'embedding', 'pooling', 'batchnorm', 
                          'skip', 'branch', 'merge']
            
            for layer_type in layer_types:
                base_color = colors.get(f'{layer_type}_fill', colors['layer_fill'])
                light_color = self._lighten_color(base_color, 0.3)
                dark_color = self._darken_color(base_color, 0.2)
                
                defs += f'''
    <linearGradient id="professional-grad-{layer_type}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{light_color};stop-opacity:1" />
      <stop offset="30%" style="stop-color:{base_color};stop-opacity:1" />
      <stop offset="100%" style="stop-color:{dark_color};stop-opacity:1" />
    </linearGradient>'''
        
        defs += '\n  </defs>'
        return defs
        
    def _render_professional_connections(self, layers: List[Layer], 
                                       architecture_graph: Dict[str, Any],
                                       positions: List[Tuple[int, int]], 
                                       colors: Dict[str, str], 
                                       styles: Dict[str, Any]) -> str:
        """Render professional connections with proper skip connection visualization."""
        connections = []
        
        # Render main pathway connections
        for i in range(len(positions) - 1):
            layer = layers[i]
            next_layer = layers[i + 1]
            
            # Skip rendering connection if next layer is a merge point (skip connection will handle it)
            if next_layer.layer_type == "merge" and hasattr(next_layer, 'merge_with') and next_layer.merge_with:
                continue
                
            # Skip rendering connection from branch (skip connection will handle the bypass)
            if layer.layer_type == "branch":
                continue
                
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            
            # Professional connection with proper styling
            start_x = x1 + styles["layer_width"] // 2
            start_y = y1
            end_x = x2 - styles["layer_width"] // 2
            end_y = y2
            
            connections.append(f'''  <path d="M {start_x},{start_y} L {end_x},{end_y}" 
        stroke="{colors['connection']}" 
        stroke-width="{styles['connection_width']}" 
        stroke-linecap="round"
        marker-end="url(#arrowhead)"
        style="filter: drop-shadow(1px 1px 1px rgba(0,0,0,0.1))"/>''')
        
        # Render skip connections with professional curved paths
        for start_idx, end_idx, branch_name in architecture_graph['skip_paths']:
            start_x, start_y = positions[start_idx]
            end_x, end_y = positions[end_idx]
            
            # Create professional curved skip connection
            skip_start_x = start_x + styles["layer_width"] // 2
            skip_start_y = start_y
            skip_end_x = end_x - styles["layer_width"] // 2
            skip_end_y = end_y
            
            # Calculate control points for smooth curve
            mid_x = (skip_start_x + skip_end_x) / 2
            curve_height = abs(skip_end_y - skip_start_y) + styles["layer_spacing_y"] // 3
            
            if skip_start_y != skip_end_y:
                # Curved skip connection
                control_y = min(skip_start_y, skip_end_y) - curve_height
                connections.append(f'''  <path d="M {skip_start_x},{skip_start_y} Q {mid_x},{control_y} {skip_end_x},{skip_end_y}" 
        stroke="{colors['connection']}" 
        stroke-width="{max(2, styles['connection_width'])}" 
        stroke-dasharray="8,4" 
        stroke-linecap="round"
        fill="none" 
        opacity="0.8"
        marker-end="url(#skip-arrowhead)"
        style="filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.2))"/>''')
            else:
                # Straight skip connection with arc
                connections.append(f'''  <path d="M {skip_start_x},{skip_start_y} Q {mid_x},{skip_start_y - 30} {skip_end_x},{skip_end_y}" 
        stroke="{colors['connection']}" 
        stroke-width="{max(2, styles['connection_width'])}" 
        stroke-dasharray="8,4" 
        stroke-linecap="round"
        fill="none" 
        opacity="0.8"
        marker-end="url(#skip-arrowhead)"
        style="filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.2))"/>''')
        
        # Add custom connections if available
        if 'custom_connections' in architecture_graph:
            custom_svg = self._render_custom_connections(
                architecture_graph['custom_connections'], layers, positions, colors, styles)
            if custom_svg:
                connections.append(custom_svg)
        
        return "\n".join(connections)
        
    def _render_professional_layer(self, layer: Layer, x: int, y: int,
                                 colors: Dict[str, str], styles: Dict[str, Any],
                                 typography: Dict[str, str]) -> str:
        """Render a layer with professional 3D styling and effects."""
        shape_info = layer.get_shape_info()
        layer_type = shape_info["type"]
        
        # Calculate rectangle position (x,y is center)
        rect_x = x - styles["layer_width"] // 2
        rect_y = y - styles["layer_height"] // 2
        
        # Check if this is NNSVG theme for advanced effects
        is_nnsvg = 'shadow' in colors
        
        if is_nnsvg and layer_type not in ["branch", "merge"]:
            # Professional 3D layer with advanced effects
            depth = styles.get('layer_depth', 12)
            
            layer_svg = f'''  <g class="professional-layer layer-{layer_type}">
    <!-- 3D depth faces -->
    <polygon points="{rect_x + styles['layer_width']},{rect_y} {rect_x + styles['layer_width'] + depth},{rect_y - depth} {rect_x + styles['layer_width'] + depth},{rect_y + styles['layer_height'] - depth} {rect_x + styles['layer_width']},{rect_y + styles['layer_height']}"
             fill="url(#professional-grad-{layer_type})"
             opacity="0.7"
             filter="url(#layer-effects)"/>
    <polygon points="{rect_x},{rect_y + styles['layer_height']} {rect_x + depth},{rect_y + styles['layer_height'] - depth} {rect_x + styles['layer_width'] + depth},{rect_y + styles['layer_height'] - depth} {rect_x + styles['layer_width']},{rect_y + styles['layer_height']}"
             fill="url(#professional-grad-{layer_type})"
             opacity="0.6"
             filter="url(#layer-effects)"/>
    
    <!-- Main face with professional gradient and effects -->
    <rect x="{rect_x}" y="{rect_y}" 
          width="{styles['layer_width']}" height="{styles['layer_height']}"
          rx="{styles['border_radius']}" ry="{styles['border_radius']}"
          fill="url(#professional-grad-{layer_type})"
          stroke="{colors['layer_stroke']}" 
          stroke-width="{styles['stroke_width']}"
          filter="url(#layer-shadow)"
          style="cursor: pointer;"/>
          
    <!-- Professional text with enhanced typography -->
    <text x="{x}" y="{y + 2}" 
          font-family="{typography['font_family']}" 
          font-size="{typography['font_size']}"
          font-weight="{typography['font_weight']}"
          text-anchor="{typography['text_anchor']}" 
          fill="{colors['text']}"
          style="text-shadow: 1px 1px 1px rgba(255,255,255,0.8); pointer-events: none;">
      {shape_info['display_text']}
    </text>
  </g>'''
        else:
            # Standard rendering for other themes or special layers
            fill_color = colors.get(f"{layer_type}_fill", colors["layer_fill"])
            layer_svg = f'''  <g class="layer layer-{layer_type}">
    <rect x="{rect_x}" y="{rect_y}" 
          width="{styles['layer_width']}" height="{styles['layer_height']}"
          rx="{styles['border_radius']}" ry="{styles['border_radius']}"
          fill="{fill_color}" stroke="{colors['layer_stroke']}" 
          stroke-width="{styles['stroke_width']}"/>
    <text x="{x}" y="{y + 5}" 
          font-family="{typography['font_family']}" 
          font-size="{typography['font_size']}"
          font-weight="{typography['font_weight']}"
          text-anchor="{typography['text_anchor']}" 
          fill="{colors['text']}">
      {shape_info['display_text']}
    </text>
  </g>'''
        
        return layer_svg
        
    def _calculate_canvas_size(self, positions: List[Tuple[float, float]], styles: Dict[str, Any]) -> Tuple[int, int]:
        """Calculate canvas size with proper margins for professional layout."""
        if not positions:
            return 400, 200
            
        max_x = max(pos[0] for pos in positions)
        max_y = max(pos[1] for pos in positions)
        min_x = min(pos[0] for pos in positions)
        min_y = min(pos[1] for pos in positions)
        
        # Add generous margins for professional appearance
        margin_x = styles["padding"] * 2
        margin_y = styles["padding"] * 2
        
        width = int(max_x - min_x + styles["layer_width"] + margin_x)
        height = int(max_y - min_y + styles["layer_height"] + margin_y)
        
        return width, height
        
    def _calculate_canvas_size_with_groups(self, positions: Dict[str, Tuple[float, float]], 
                                         groups: List, styles: Dict[str, Any]) -> Tuple[int, int]:
        """Calculate canvas size including space for group bounding boxes."""
        # Convert positions dict to list of coordinate tuples
        position_list = list(positions.values())
        base_width, base_height = self._calculate_canvas_size(position_list, styles)
        
        # Add extra space for group labels and padding
        if groups:
            group_padding = styles.get("group_padding", 30)
            base_width += group_padding * 2
            base_height += group_padding * 2
            
        return base_width, base_height
        
    def _render_layer_groups(self, groups: List, layers: List[Layer], 
                           positions: List[Tuple[int, int]], colors: Dict[str, str], 
                           styles: Dict[str, Any]) -> str:
        """Render visual bounding boxes for layer groups."""
        group_svg_parts = []
        
        for group in groups:
            if not group.layers:
                continue
                
            # Find positions of layers in this group
            group_positions = []
            for layer in layers:
                if layer.name in group.layers and layer.name in positions:
                    group_positions.append(positions[layer.name])
                    
            if not group_positions:
                continue
                
            # Calculate bounding box for the group
            min_x = min(pos[0] for pos in group_positions) - styles["layer_width"] // 2
            max_x = max(pos[0] for pos in group_positions) + styles["layer_width"] // 2
            min_y = min(pos[1] for pos in group_positions) - styles["layer_height"] // 2
            max_y = max(pos[1] for pos in group_positions) + styles["layer_height"] // 2
            
            # Add padding
            padding = styles.get("group_padding", 20)
            min_x -= padding
            max_x += padding
            min_y -= padding
            max_y += padding
            
            # Get group styling
            group_fill = group.style.get("fill", "#f8f9fa")
            group_stroke = group.style.get("stroke", "#dee2e6")
            group_stroke_width = group.style.get("stroke-width", "2")
            group_opacity = group.style.get("opacity", "0.3")
            
            # Render group bounding box
            group_svg_parts.append(f'''  <g class="layer-group" id="group-{group.name}">
    <rect x="{min_x}" y="{min_y}" width="{max_x - min_x}" height="{max_y - min_y}"
          fill="{group_fill}" stroke="{group_stroke}" 
          stroke-width="{group_stroke_width}" opacity="{group_opacity}"
          rx="8" ry="8" stroke-dasharray="5,5"/>
    <text x="{min_x + 10}" y="{min_y - 5}" 
          font-family="Source Sans Pro, Arial, sans-serif" 
          font-size="12px" font-weight="600"
          fill="{colors['text']}">{group.name}</text>
  </g>''')
        
        return "\n".join(group_svg_parts)
        
    def _render_custom_connections(self, custom_connections: List, layers: List[Layer],
                                 positions: List[Tuple[int, int]], colors: Dict[str, str],
                                 styles: Dict[str, Any]) -> str:
        """Render custom connections between named layers."""
        connection_svg_parts = []
        
        # Create a mapping from layer names to positions
        name_to_position = {}
        for i, layer in enumerate(layers):
            name_to_position[layer.name] = positions[i]
            
        for connection in custom_connections:
            if connection.source_name not in name_to_position or connection.target_name not in name_to_position:
                continue  # Skip connections to non-existent layers
                
            source_x, source_y = name_to_position[connection.source_name]
            target_x, target_y = name_to_position[connection.target_name]
            
            # Calculate connection points
            start_x = source_x + styles["layer_width"] // 2
            start_y = source_y
            end_x = target_x - styles["layer_width"] // 2
            end_y = target_y
            
            # Style based on connection type
            if connection.style == "skip":
                # Curved skip connection
                mid_x = (start_x + end_x) / 2
                control_y = min(start_y, end_y) - 40
                
                connection_svg_parts.append(f'''  <path d="M {start_x},{start_y} Q {mid_x},{control_y} {end_x},{end_y}" 
        stroke="{colors.get('skip_connection', colors['connection'])}" 
        stroke-width="{styles['connection_width'] + 1}" 
        stroke-dasharray="6,4" 
        stroke-linecap="round"
        fill="none" 
        opacity="0.8"
        marker-end="url(#skip-arrowhead)"/>''')
            elif connection.style == "attention":
                # Dotted attention connection
                connection_svg_parts.append(f'''  <path d="M {start_x},{start_y} L {end_x},{end_y}" 
        stroke="{colors.get('attention_connection', '#7e57c2')}" 
        stroke-width="{styles['connection_width']}" 
        stroke-dasharray="2,3" 
        stroke-linecap="round"
        marker-end="url(#arrowhead)"/>''')
            else:
                # Standard connection
                connection_svg_parts.append(f'''  <path d="M {start_x},{start_y} L {end_x},{end_y}" 
        stroke="{colors['connection']}" 
        stroke-width="{styles['connection_width']}" 
        stroke-linecap="round"
        marker-end="url(#arrowhead)"/>''')
        
        return "\n".join(connection_svg_parts)
        
    def _render_advanced_connections(self, diagram, layout_engine: AdvancedLayoutEngine,
                                   positions: Dict[str, Tuple[float, float]], 
                                   colors: Dict[str, str], styles: Dict[str, Any]) -> str:
        """Render connections using the advanced layout engine."""
        connection_svg_parts = []
        
        if not layout_engine.graph:
            return ""
        
        # Render all edges in the graph
        for source_node in layout_engine.graph.nodes:
            for target_node in layout_engine.graph.get_successors(source_node):
                # Get edge path (may include dummy nodes)
                path = layout_engine.get_edge_path(source_node, target_node)
                
                if len(path) < 2:
                    continue
                
                # Determine connection style
                style = "regular"
                if hasattr(diagram, 'connections'):
                    for conn in diagram.connections:
                        if ((conn.source_name == source_node and conn.target_name == target_node) or
                            self._is_connection_path(conn.source_name, conn.target_name, 
                                                   source_node, target_node, layout_engine)):
                            style = conn.style
                            break
                
                # Check if this is a skip connection based on ranks
                source_rank = layout_engine.ranks.get(source_node, 0)
                target_rank = layout_engine.ranks.get(target_node, 0)
                if target_rank - source_rank > 1:
                    style = "skip"
                
                # Render the connection
                if len(path) == 2:
                    # Direct connection
                    start_x, start_y = path[0]
                    end_x, end_y = path[1]
                    
                    if style == "skip":
                        # Curved skip connection
                        mid_x = (start_x + end_x) / 2
                        control_y = start_y - 30
                        
                        connection_svg_parts.append(f'''  <path d="M {start_x},{start_y} Q {mid_x},{control_y} {end_x},{end_y}" 
        stroke="{colors.get('skip_connection', colors['connection'])}" 
        stroke-width="{styles['connection_width'] + 1}" 
        stroke-dasharray="6,4" 
        stroke-linecap="round"
        fill="none" 
        opacity="0.8"
        marker-end="url(#skip-arrowhead)"/>''')
                    elif style == "attention":
                        # Dotted attention connection  
                        connection_svg_parts.append(f'''  <path d="M {start_x},{start_y} L {end_x},{end_y}" 
        stroke="{colors.get('attention_connection', '#7e57c2')}" 
        stroke-width="{styles['connection_width']}" 
        stroke-dasharray="2,3" 
        stroke-linecap="round"
        marker-end="url(#arrowhead)"/>''')
                    else:
                        # Regular connection
                        connection_svg_parts.append(f'''  <path d="M {start_x},{start_y} L {end_x},{end_y}" 
        stroke="{colors['connection']}" 
        stroke-width="{styles['connection_width']}" 
        stroke-linecap="round"
        marker-end="url(#arrowhead)"/>''')
                else:
                    # Multi-segment connection through dummy nodes
                    path_str = f"M {path[0][0]},{path[0][1]}"
                    for i in range(1, len(path)):
                        path_str += f" L {path[i][0]},{path[i][1]}"
                    
                    connection_svg_parts.append(f'''  <path d="{path_str}" 
        stroke="{colors['connection']}" 
        stroke-width="{styles['connection_width']}" 
        stroke-linecap="round"
        fill="none"
        marker-end="url(#arrowhead)"/>''')
        
        return "\n".join(connection_svg_parts)
    
    def _is_connection_path(self, orig_source: str, orig_target: str, 
                          current_source: str, current_target: str, 
                          layout_engine: AdvancedLayoutEngine) -> bool:
        """Check if current edge is part of an original connection path."""
        # This would check if the current edge is part of the path from
        # orig_source to orig_target, potentially through dummy nodes
        # For now, simple implementation
        return (orig_source == current_source and orig_target == current_target)
        
    def _empty_svg(self, theme: Theme) -> str:
        """Generate professional empty SVG."""
        colors = theme.get_colors()
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" viewBox="0 0 400 200"
     style="font-family: 'Source Sans Pro', Arial, sans-serif;">
  <rect width="100%" height="100%" fill="{colors['background']}"/>
  <text x="200" y="100" text-anchor="middle" 
        font-family="Source Sans Pro, Arial, sans-serif" 
        font-size="16" font-weight="300"
        fill="{colors['text']}">
    Empty Diagram
  </text>
</svg>'''

    def _lighten_color(self, hex_color: str, factor: float) -> str:
        """Lighten a hex color by a given factor."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def _darken_color(self, hex_color: str, factor: float) -> str:
        """Darken a hex color by a given factor."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        r = max(0, int(r * (1 - factor)))
        g = max(0, int(g * (1 - factor)))
        b = max(0, int(b * (1 - factor)))
        
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def set_layout(self, layout: str) -> None:
        """Set the layout type."""
        if layout in ["horizontal", "vertical"]:
            self.layout = layout
        else:
            raise ValueError(f"Unsupported layout: {layout}")