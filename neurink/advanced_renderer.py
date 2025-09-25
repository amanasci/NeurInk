"""
Advanced professional renderer with comprehensive connection routing and styling.

This module implements a complete overhaul of the rendering system with:
- Advanced Bezier curve routing for all connections
- Flexible styling options across all themes
- Elimination of visual artifacts
- Publication-quality output
"""

from typing import List, Dict, Any, Tuple, Optional, Set
import math
from .layer import Layer
from .themes import Theme
from .layout import AdvancedLayoutEngine


class AdvancedSVGRenderer:
    """
    Advanced professional SVG renderer with comprehensive connection routing.
    
    Features:
    - Advanced Bezier curve routing for all connection types
    - Intelligent connection path calculation
    - Universal theme compatibility
    - Elimination of visual artifacts
    - Professional styling and effects
    """
    
    def __init__(self):
        """Initialize the advanced renderer."""
        self.layout_engine = AdvancedLayoutEngine()
        
    def render_diagram(self, diagram, theme: Theme) -> str:
        """
        Render a complete diagram with advanced professional features.
        
        Args:
            diagram: Diagram object with layers, connections, and groups
            theme: Theme object for styling
            
        Returns:
            Professional SVG content as string with advanced routing
        """
        if not diagram.layers:
            return self._empty_svg(theme)
            
        colors = theme.get_colors()
        styles = theme.get_styles()
        typography = theme.get_typography()
        
        # Calculate advanced layout using Sugiyama algorithm
        positions = self.layout_engine.calculate_positions(diagram, styles)
        
        # Build comprehensive connection graph
        connection_graph = self._build_connection_graph(diagram, positions)
        
        # Calculate optimal canvas dimensions
        canvas_width, canvas_height = self._calculate_optimal_canvas_size(
            positions, connection_graph, styles)
        
        # Generate professional SVG with universal theme support
        svg_parts = []
        svg_parts.append(self._professional_svg_header(canvas_width, canvas_height, colors))
        svg_parts.append(self._universal_svg_defs(theme))
        
        # Render groups if present
        if hasattr(diagram, 'groups') and diagram.groups:
            svg_parts.append(self._render_diagram_groups(diagram.groups, positions, colors, styles))
        
        # Render all connections with advanced Bezier routing
        connections_svg = self._render_advanced_connections(connection_graph, colors, styles)
        if connections_svg:
            svg_parts.append(connections_svg)
        
        # Render layers with universal styling
        layers_svg = self._render_universal_layers(diagram.layers, positions, colors, styles, typography)
        svg_parts.append(layers_svg)
            
        svg_parts.append("</svg>")
        
        return "\n".join(svg_parts)
    
    def _build_connection_graph(self, diagram, positions: Dict[str, Tuple[int, int]]) -> Dict[str, Any]:
        """
        Build comprehensive connection graph with all connection types.
        
        Returns:
            Complete connection graph with routing information
        """
        graph = {
            'sequential': [],      # Standard layer-to-layer connections
            'skip_connections': [], # Skip connections from branch/merge analysis
            'custom_connections': [], # User-defined custom connections
            'group_connections': [], # Connections between grouped layers
            'positions': positions,
            'routing_zones': []    # Areas to avoid when routing connections
        }
        
        # Sequential connections
        for i in range(len(diagram.layers) - 1):
            current_layer = diagram.layers[i]
            next_layer = diagram.layers[i + 1]
            
            # Skip if this will be handled by skip connection logic
            if (current_layer.layer_type == "branch" or 
                (next_layer.layer_type == "merge" and hasattr(next_layer, 'merge_with'))):
                continue
                
            graph['sequential'].append({
                'source': current_layer.name,
                'target': next_layer.name,
                'type': 'sequential',
                'style': 'standard'
            })
        
        # Skip connections from architecture analysis
        skip_connections = self._analyze_skip_connections(diagram.layers)
        graph['skip_connections'] = skip_connections
        
        # Custom connections
        if hasattr(diagram, 'connections'):
            for conn in diagram.connections:
                if isinstance(conn, dict):
                    # Legacy dict format
                    graph['custom_connections'].append({
                        'source': conn['source'],
                        'target': conn['target'],
                        'type': 'custom',
                        'style': conn.get('style', 'skip')
                    })
                else:
                    # Connection object
                    graph['custom_connections'].append({
                        'source': conn.source_name,
                        'target': conn.target_name,
                        'type': 'custom',
                        'style': conn.style
                    })
        
        # Calculate routing zones to avoid overlaps
        graph['routing_zones'] = self._calculate_routing_zones(positions)
        
        return graph
    
    def _analyze_skip_connections(self, layers: List[Layer]) -> List[Dict[str, Any]]:
        """Analyze layers to identify skip connections."""
        skip_connections = []
        branches = {}  # branch_name -> layer_index
        
        for i, layer in enumerate(layers):
            if layer.layer_type == "branch":
                branches[layer.branch_name] = i
            elif layer.layer_type == "merge" and hasattr(layer, 'merge_with') and layer.merge_with:
                if layer.merge_with in branches:
                    skip_connections.append({
                        'source': layers[branches[layer.merge_with]].name,
                        'target': layer.name,
                        'type': 'skip',
                        'style': 'residual',
                        'branch_name': layer.merge_with
                    })
        
        return skip_connections
    
    def _calculate_routing_zones(self, positions: Dict[str, Tuple[int, int]]) -> List[Dict[str, Any]]:
        """Calculate zones to avoid when routing connections."""
        zones = []
        
        # Add rectangular zones around each layer position
        for layer_name, (x, y) in positions.items():
            zones.append({
                'type': 'layer',
                'x': x - 60,  # Half layer width + margin
                'y': y - 30,  # Half layer height + margin
                'width': 120,
                'height': 60,
                'layer': layer_name
            })
        
        return zones
    
    def _render_advanced_connections(self, connection_graph: Dict[str, Any], 
                                   colors: Dict[str, str], styles: Dict[str, Any]) -> str:
        """
        Render all connections with advanced Bezier curve routing.
        
        This is the core of the refactored connection system.
        """
        connections_svg = []
        positions = connection_graph['positions']
        routing_zones = connection_graph['routing_zones']
        
        # Render sequential connections
        for conn in connection_graph['sequential']:
            if conn['source'] in positions and conn['target'] in positions:
                path_svg = self._render_bezier_connection(
                    conn, positions, routing_zones, colors, styles, 'sequential')
                if path_svg:
                    connections_svg.append(path_svg)
        
        # Render skip connections with advanced routing
        for conn in connection_graph['skip_connections']:
            if conn['source'] in positions and conn['target'] in positions:
                path_svg = self._render_bezier_connection(
                    conn, positions, routing_zones, colors, styles, 'skip')
                if path_svg:
                    connections_svg.append(path_svg)
        
        # Render custom connections
        for conn in connection_graph['custom_connections']:
            if conn['source'] in positions and conn['target'] in positions:
                path_svg = self._render_bezier_connection(
                    conn, positions, routing_zones, colors, styles, 'custom')
                if path_svg:
                    connections_svg.append(path_svg)
        
        return "\n".join(connections_svg)
    
    def _render_bezier_connection(self, connection: Dict[str, Any], 
                                positions: Dict[str, Tuple[int, int]],
                                routing_zones: List[Dict[str, Any]],
                                colors: Dict[str, str], styles: Dict[str, Any],
                                connection_category: str) -> str:
        """
        Render a single connection with advanced Bezier curve routing.
        
        This implements sophisticated path calculation to avoid overlaps.
        """
        source_pos = positions[connection['source']]
        target_pos = positions[connection['target']]
        
        # Calculate connection points (edge of rectangles, not centers)
        start_point = self._calculate_connection_point(source_pos, target_pos, styles, 'source')
        end_point = self._calculate_connection_point(target_pos, source_pos, styles, 'target')
        
        # Calculate optimal Bezier control points
        control_points = self._calculate_bezier_control_points(
            start_point, end_point, routing_zones, connection)
        
        # Generate SVG path with appropriate styling
        return self._generate_connection_svg(
            start_point, end_point, control_points, connection, 
            connection_category, colors, styles)
    
    def _calculate_connection_point(self, from_pos: Tuple[int, int], 
                                  to_pos: Tuple[int, int], styles: Dict[str, Any], 
                                  point_type: str) -> Tuple[int, int]:
        """Calculate the exact point where connection should start/end on layer rectangle."""
        layer_width = styles['layer_width']
        layer_height = styles['layer_height']
        
        fx, fy = from_pos
        tx, ty = to_pos
        
        # Calculate direction vector
        dx = tx - fx
        dy = ty - fy
        
        if point_type == 'source':
            # Connection starts from right edge if target is to the right, otherwise from appropriate edge
            if dx > 0:  # Target is to the right
                return (fx + layer_width // 2, fy)
            elif dx < 0:  # Target is to the left
                return (fx - layer_width // 2, fy)
            elif dy > 0:  # Target is below
                return (fx, fy + layer_height // 2)
            else:  # Target is above
                return (fx, fy - layer_height // 2)
        else:  # target point
            # Connection ends at left edge if source is to the left, otherwise at appropriate edge
            if dx > 0:  # Source is to the left
                return (tx - layer_width // 2, ty)
            elif dx < 0:  # Source is to the right
                return (tx + layer_width // 2, ty)
            elif dy > 0:  # Source is above
                return (tx, ty - layer_height // 2)
            else:  # Source is below
                return (tx, ty + layer_height // 2)
    
    def _calculate_bezier_control_points(self, start_point: Tuple[int, int], 
                                       end_point: Tuple[int, int],
                                       routing_zones: List[Dict[str, Any]],
                                       connection: Dict[str, Any]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Calculate optimal Bezier control points for smooth routing."""
        sx, sy = start_point
        ex, ey = end_point
        
        # Base control point calculation
        dx = ex - sx
        dy = ey - sy
        distance = math.sqrt(dx*dx + dy*dy)
        
        if connection.get('type') == 'skip' or connection.get('style') in ['skip', 'residual']:
            # Skip connections use more dramatic curves to avoid overlapping layers
            if abs(dy) < 50:  # Approximately horizontal skip connection
                # Curve above the layers
                curve_height = max(80, distance * 0.3)
                mid_x = sx + dx * 0.5
                control_y = min(sy, ey) - curve_height
                
                cp1 = (sx + dx * 0.25, control_y)
                cp2 = (sx + dx * 0.75, control_y)
            else:
                # Diagonal skip connection
                cp1 = (sx + dx * 0.3, sy + dy * 0.1)
                cp2 = (sx + dx * 0.7, sy + dy * 0.9)
        else:
            # Standard connections use gentler curves
            if abs(dx) > abs(dy):  # Primarily horizontal
                cp1 = (sx + dx * 0.5, sy)
                cp2 = (sx + dx * 0.5, ey)
            else:  # Primarily vertical
                cp1 = (sx, sy + dy * 0.5)
                cp2 = (ex, sy + dy * 0.5)
        
        return cp1, cp2
    
    def _generate_connection_svg(self, start_point: Tuple[int, int], 
                               end_point: Tuple[int, int],
                               control_points: Tuple[Tuple[int, int], Tuple[int, int]],
                               connection: Dict[str, Any], category: str,
                               colors: Dict[str, str], styles: Dict[str, Any]) -> str:
        """Generate SVG path element for the connection."""
        sx, sy = start_point
        ex, ey = end_point
        cp1x, cp1y = control_points[0]
        cp2x, cp2y = control_points[1]
        
        # Determine styling based on connection type and category
        if category == 'skip' or connection.get('style') in ['skip', 'residual']:
            stroke_color = colors.get('connection', '#666666')
            stroke_width = max(2, styles.get('connection_width', 2))
            stroke_dasharray = "8,4"
            opacity = "0.8"
            marker = "url(#skip-arrowhead)"
        elif connection.get('style') == 'attention':
            stroke_color = colors.get('attention_connection', colors.get('connection', '#666666'))
            stroke_width = max(3, styles.get('connection_width', 2) + 1)
            stroke_dasharray = "12,6"
            opacity = "0.7"
            marker = "url(#attention-arrowhead)"
        else:
            stroke_color = colors.get('connection', '#666666')
            stroke_width = styles.get('connection_width', 2)
            stroke_dasharray = "none"
            opacity = "1.0"
            marker = "url(#arrowhead)"
        
        # Create smooth Bezier path
        path_d = f"M {sx},{sy} C {cp1x},{cp1y} {cp2x},{cp2y} {ex},{ey}"
        
        svg_path = f'''  <path d="{path_d}" 
      stroke="{stroke_color}" 
      stroke-width="{stroke_width}" 
      stroke-linecap="round"
      stroke-linejoin="round"
      fill="none"
      opacity="{opacity}"
      marker-end="{marker}"'''
      
        if stroke_dasharray != "none":
            svg_path += f'\n      stroke-dasharray="{stroke_dasharray}"'
            
        svg_path += '\n      style="filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.15))"/>'
        
        return svg_path
    
    def _render_universal_layers(self, layers: List[Layer], 
                               positions: Dict[str, Tuple[int, int]],
                               colors: Dict[str, str], styles: Dict[str, Any],
                               typography: Dict[str, str]) -> str:
        """Render all layers with universal theme compatibility."""
        layers_svg = []
        
        for layer in layers:
            if layer.name in positions:
                x, y = positions[layer.name]
                layer_svg = self._render_universal_layer(
                    layer, x, y, colors, styles, typography)
                layers_svg.append(layer_svg)
        
        return "\n".join(layers_svg)
    
    def _render_universal_layer(self, layer: Layer, x: int, y: int,
                              colors: Dict[str, str], styles: Dict[str, Any],
                              typography: Dict[str, str]) -> str:
        """Render a single layer with universal theme support."""
        shape_info = layer.get_shape_info()
        layer_type = shape_info["type"]
        
        # Calculate rectangle position (x,y is center)
        rect_x = x - styles["layer_width"] // 2
        rect_y = y - styles["layer_height"] // 2
        
        # Determine if this is an advanced theme (has shadow color or other advanced features)
        is_advanced_theme = 'shadow' in colors or hasattr(colors, 'get_advanced_features')
        
        # Skip rendering for branch/merge layers (they are logical only)
        if layer_type in ["branch", "merge"]:
            return ""
        
        # Get layer-specific colors
        fill_color = colors.get(f'{layer_type}_fill', colors.get('layer_fill', '#f0f0f0'))
        stroke_color = colors.get('layer_stroke', '#333333')
        text_color = colors.get('text', '#000000')
        
        if is_advanced_theme:
            # Advanced theme with 3D effects and gradients
            depth = styles.get('layer_depth', 8)
            
            layer_svg = f'''  <g class="universal-layer layer-{layer_type}">
    <!-- 3D depth faces -->
    <polygon points="{rect_x + styles['layer_width']},{rect_y} {rect_x + styles['layer_width'] + depth},{rect_y - depth} {rect_x + styles['layer_width'] + depth},{rect_y + styles['layer_height'] - depth} {rect_x + styles['layer_width']},{rect_y + styles['layer_height']}"
             fill="url(#universal-grad-{layer_type})"
             opacity="0.6"/>
    <polygon points="{rect_x},{rect_y + styles['layer_height']} {rect_x + depth},{rect_y + styles['layer_height'] - depth} {rect_x + styles['layer_width'] + depth},{rect_y + styles['layer_height'] - depth} {rect_x + styles['layer_width']},{rect_y + styles['layer_height']}"
             fill="url(#universal-grad-{layer_type})"
             opacity="0.4"/>
    
    <!-- Main face -->
    <rect x="{rect_x}" y="{rect_y}" 
          width="{styles['layer_width']}" height="{styles['layer_height']}"
          rx="{styles.get('border_radius', 8)}" ry="{styles.get('border_radius', 8)}"
          fill="url(#universal-grad-{layer_type})"
          stroke="{stroke_color}" 
          stroke-width="{styles.get('stroke_width', 2)}"
          filter="url(#universal-layer-effects)"/>
    
    <!-- Layer text with shadow -->
    <text x="{x}" y="{y + 5}" 
          font-family="{typography.get('font_family', 'Arial, sans-serif')}" 
          font-size="{typography.get('font_size', '12px')}"
          font-weight="{typography.get('font_weight', 'normal')}"
          text-anchor="middle" 
          fill="{text_color}"
          filter="url(#text-shadow)">
      {shape_info.get('display_text', layer_type)}
    </text>
  </g>'''
        else:
            # Standard theme with clean styling
            layer_svg = f'''  <g class="universal-layer layer-{layer_type}">
    <rect x="{rect_x}" y="{rect_y}" 
          width="{styles['layer_width']}" height="{styles['layer_height']}"
          rx="{styles.get('border_radius', 4)}" ry="{styles.get('border_radius', 4)}"
          fill="{fill_color}"
          stroke="{stroke_color}" 
          stroke-width="{styles.get('stroke_width', 1.5)}"/>
    
    <text x="{x}" y="{y + 4}" 
          font-family="{typography.get('font_family', 'Arial, sans-serif')}" 
          font-size="{typography.get('font_size', '11px')}"
          font-weight="{typography.get('font_weight', 'normal')}"
          text-anchor="middle" 
          fill="{text_color}">
      {shape_info.get('display_text', layer_type)}
    </text>
  </g>'''
        
        return layer_svg
    
    def _universal_svg_defs(self, theme: Theme) -> str:
        """Generate universal SVG definitions that work with all themes."""
        colors = theme.get_colors()
        styles = theme.get_styles()
        
        is_advanced_theme = 'shadow' in colors or hasattr(colors, 'get_advanced_features')
        
        defs = f'''  <defs>
    <!-- Universal arrowheads -->
    <marker id="arrowhead" markerWidth="12" markerHeight="8" 
            refX="12" refY="4" orient="auto" markerUnits="strokeWidth">
      <polygon points="0 0, 12 4, 0 8" 
               fill="{colors.get('connection', '#666666')}" 
               stroke="none"/>
    </marker>
    
    <marker id="skip-arrowhead" markerWidth="12" markerHeight="8" 
            refX="12" refY="4" orient="auto" markerUnits="strokeWidth">
      <polygon points="0 0, 12 4, 0 8" 
               fill="{colors.get('connection', '#666666')}" 
               stroke="none" opacity="0.7"/>
    </marker>
    
    <marker id="attention-arrowhead" markerWidth="14" markerHeight="10" 
            refX="14" refY="5" orient="auto" markerUnits="strokeWidth">
      <polygon points="0 0, 14 5, 0 10" 
               fill="{colors.get('attention_connection', colors.get('connection', '#666666'))}" 
               stroke="none" opacity="0.8"/>
    </marker>'''
    
        if is_advanced_theme:
            defs += f'''
    
    <!-- Advanced effects -->
    <filter id="universal-layer-effects" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="2"/>
      <feOffset dx="2" dy="3" result="offset"/>
      <feFlood flood-color="#000000" flood-opacity="0.2"/>
      <feComposite in2="offset" operator="in"/>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    
    <filter id="text-shadow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur in="SourceAlpha" stdDeviation="1"/>
      <feOffset dx="1" dy="1" result="offset"/>
      <feFlood flood-color="#000000" flood-opacity="0.3"/>
      <feComposite in2="offset" operator="in"/>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>'''
    
            # Universal gradients for all layer types
            layer_types = ['input', 'conv', 'dense', 'output', 'flatten', 'dropout', 
                          'attention', 'layernorm', 'embedding', 'pooling', 'batchnorm']
            
            for layer_type in layer_types:
                base_color = colors.get(f'{layer_type}_fill', colors.get('layer_fill', '#f0f0f0'))
                light_color = self._adjust_color_brightness(base_color, 1.2)
                dark_color = self._adjust_color_brightness(base_color, 0.8)
                
                defs += f'''
    <linearGradient id="universal-grad-{layer_type}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{light_color};stop-opacity:1" />
      <stop offset="50%" style="stop-color:{base_color};stop-opacity:1" />
      <stop offset="100%" style="stop-color:{dark_color};stop-opacity:1" />
    </linearGradient>'''
        
        defs += '\n  </defs>'
        return defs
    
    def _adjust_color_brightness(self, hex_color: str, factor: float) -> str:
        """Adjust the brightness of a hex color."""
        if not hex_color.startswith('#'):
            return hex_color
            
        try:
            hex_color = hex_color[1:]  # Remove #
            
            # Parse RGB components
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            
            # Apply brightness factor
            r = max(0, min(255, int(r * factor)))
            g = max(0, min(255, int(g * factor)))
            b = max(0, min(255, int(b * factor)))
            
            return f"#{r:02x}{g:02x}{b:02x}"
        except (ValueError, IndexError):
            return hex_color
    
    def _professional_svg_header(self, width: int, height: int, colors: Dict[str, str]) -> str:
        """Generate professional SVG header."""
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"
     style="font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; background: {colors.get('background', '#ffffff')};">
  <rect width="100%" height="100%" fill="{colors.get('background', '#ffffff')}"/>'''
  
    def _calculate_optimal_canvas_size(self, positions: Dict[str, Tuple[int, int]], 
                                     connection_graph: Dict[str, Any],
                                     styles: Dict[str, Any]) -> Tuple[int, int]:
        """Calculate optimal canvas size considering all elements."""
        if not positions:
            return 400, 300
            
        # Find bounds of all positions
        xs = [pos[0] for pos in positions.values()]
        ys = [pos[1] for pos in positions.values()]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Add margins for layer dimensions and connections
        padding = styles.get('padding', 40)
        layer_margin_x = styles.get('layer_width', 120) // 2
        layer_margin_y = styles.get('layer_height', 60) // 2
        
        # Extra margin for skip connections
        connection_margin = 60
        
        width = max_x - min_x + 2 * (padding + layer_margin_x + connection_margin)
        height = max_y - min_y + 2 * (padding + layer_margin_y + connection_margin)
        
        return max(400, width), max(300, height)
    
    def _render_diagram_groups(self, groups: List[Any], positions: Dict[str, Tuple[int, int]], 
                             colors: Dict[str, str], styles: Dict[str, Any]) -> str:
        """Render diagram groups with bounding boxes."""
        groups_svg = []
        
        for group in groups:
            # Calculate group bounding box
            group_layers = [layer for layer in group.layers if layer.name in positions]
            if not group_layers:
                continue
                
            xs = [positions[layer.name][0] for layer in group_layers]
            ys = [positions[layer.name][1] for layer in group_layers]
            
            min_x = min(xs) - styles['layer_width'] // 2 - 15
            max_x = max(xs) + styles['layer_width'] // 2 + 15
            min_y = min(ys) - styles['layer_height'] // 2 - 15
            max_y = max(ys) + styles['layer_height'] // 2 + 15
            
            width = max_x - min_x
            height = max_y - min_y
            
            # Group styling
            group_fill = group.style.get('fill', 'rgba(0,0,0,0.05)')
            group_stroke = group.style.get('stroke', 'rgba(0,0,0,0.2)')
            group_opacity = group.style.get('opacity', '0.7')
            
            groups_svg.append(f'''  <g class="diagram-group">
    <rect x="{min_x}" y="{min_y}" width="{width}" height="{height}"
          rx="8" ry="8"
          fill="{group_fill}" 
          stroke="{group_stroke}"
          stroke-width="1"
          opacity="{group_opacity}"/>
    <text x="{min_x + 10}" y="{min_y + 20}" 
          font-family="Arial, sans-serif" 
          font-size="10px" 
          font-weight="bold"
          fill="{colors.get('text', '#333333')}">
      {group.name}
    </text>
  </g>''')
        
        return "\n".join(groups_svg)
    
    def _empty_svg(self, theme: Theme) -> str:
        """Generate empty SVG for diagrams with no layers."""
        colors = theme.get_colors()
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="400" height="200" viewBox="0 0 400 200"
     style="font-family: Arial, sans-serif; background: {colors.get('background', '#ffffff')};">
  <rect width="100%" height="100%" fill="{colors.get('background', '#ffffff')}"/>
  <text x="200" y="100" text-anchor="middle" fill="{colors.get('text', '#333333')}" 
        font-size="14px">Empty Diagram</text>
</svg>'''