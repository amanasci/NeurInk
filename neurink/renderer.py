"""
SVG renderer for neural network diagrams.

Converts layer definitions to clean, scalable SVG output with
theme support and automatic layout using Graphviz.
"""

from typing import List, Dict, Any, Tuple
import networkx as nx
import graphviz
from .layer import Layer
from .themes import Theme


class SVGRenderer:
    """Renders neural network diagrams to SVG format."""
    
    def __init__(self):
        """Initialize the SVG renderer."""
        self.layout = "horizontal"  # horizontal, vertical, custom
        
    def render(self, layers: List[Layer], theme: Theme) -> str:
        """
        Render layers to SVG string.
        
        Args:
            layers: List of layer objects to render
            theme: Theme object for styling
            
        Returns:
            SVG content as string
        """
        if not layers:
            return self._empty_svg(theme)
            
        colors = theme.get_colors()
        styles = theme.get_styles() 
        typography = theme.get_typography()
        
        # Calculate layout positions
        positions = self._calculate_positions(layers, styles)
        
        # Calculate canvas size
        canvas_width, canvas_height = self._calculate_canvas_size(positions, styles)
        
        # Generate SVG content
        svg_parts = []
        svg_parts.append(self._svg_header(canvas_width, canvas_height, colors["background"]))
        svg_parts.append(self._svg_defs(theme))
        
        # Render connections between layers
        if len(layers) > 1:
            svg_parts.append(self._render_connections(positions, colors, styles))
        
        # Render layers
        for i, layer in enumerate(layers):
            x, y = positions[i]
            svg_parts.append(self._render_layer(layer, x, y, colors, styles, typography))
            
        svg_parts.append("</svg>")
        
        return "\n".join(svg_parts)
        
    def _empty_svg(self, theme: Theme) -> str:
        """Generate empty SVG for diagrams with no layers."""
        colors = theme.get_colors()
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100" viewBox="0 0 200 100">
  <rect width="100%" height="100%" fill="{colors['background']}"/>
  <text x="100" y="50" text-anchor="middle" font-family="Arial" font-size="14" fill="{colors['text']}">
    Empty Diagram
  </text>
</svg>'''
        
    def _svg_header(self, width: int, height: int, background: str) -> str:
        """Generate SVG header with dimensions."""
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="{background}"/>'''
        
    def _svg_defs(self, theme: Theme) -> str:
        """Generate SVG definitions for reusable elements."""
        colors = theme.get_colors()
        return f'''  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" 
            refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{colors['connection']}" />
    </marker>
  </defs>'''
        
    def _calculate_positions(self, layers: List[Layer], styles: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Calculate x,y positions for each layer."""
        positions = []
        
        if self.layout == "horizontal":
            # Ensure first layer starts at padding + half layer width to avoid negative coordinates
            x_start = styles["padding"] + styles["layer_width"] // 2
            y_center = styles["padding"] + styles["layer_height"] // 2
            
            for i in range(len(layers)):
                x = x_start + i * styles["layer_spacing_x"]
                y = y_center
                positions.append((x, y))
                
        elif self.layout == "vertical":
            x_center = styles["padding"] + styles["layer_width"] // 2
            # Ensure first layer starts at padding + half layer height to avoid negative coordinates
            y_start = styles["padding"] + styles["layer_height"] // 2
            
            for i in range(len(layers)):
                x = x_center
                y = y_start + i * styles["layer_spacing_y"]
                positions.append((x, y))
                
        return positions
        
    def _calculate_canvas_size(self, positions: List[Tuple[int, int]], styles: Dict[str, Any]) -> Tuple[int, int]:
        """Calculate required canvas dimensions."""
        if not positions:
            return 200, 100
            
        max_x = max(pos[0] for pos in positions)
        max_y = max(pos[1] for pos in positions)
        
        width = max_x + styles["layer_width"] + styles["padding"]
        height = max_y + styles["layer_height"] + styles["padding"]
        
        return width, height
        
    def _render_connections(self, positions: List[Tuple[int, int]], 
                          colors: Dict[str, str], styles: Dict[str, Any]) -> str:
        """Render connections between layers."""
        connections = []
        
        for i in range(len(positions) - 1):
            x1, y1 = positions[i]
            x2, y2 = positions[i + 1]
            
            if self.layout == "horizontal":
                # Connect from right edge of first layer to left edge of second layer
                start_x = x1 + styles["layer_width"] // 2
                start_y = y1
                end_x = x2 - styles["layer_width"] // 2
                end_y = y2
            else:  # vertical
                # Connect from bottom edge of first layer to top edge of second layer
                start_x = x1
                start_y = y1 + styles["layer_height"] // 2
                end_x = x2
                end_y = y2 - styles["layer_height"] // 2
            
            connections.append(f'''  <line x1="{start_x}" y1="{start_y}" x2="{end_x}" y2="{end_y}"
        stroke="{colors['connection']}" stroke-width="{styles['connection_width']}"
        marker-end="url(#arrowhead)"/>''')
            
        return "\n".join(connections)
        
    def _render_layer(self, layer: Layer, x: int, y: int, 
                     colors: Dict[str, str], styles: Dict[str, Any], 
                     typography: Dict[str, str]) -> str:
        """Render a single layer."""
        shape_info = layer.get_shape_info()
        layer_type = shape_info["type"]
        
        # Get layer-specific fill color
        fill_color = colors.get(f"{layer_type}_fill", colors["layer_fill"])
        
        # Calculate rectangle position (x,y is center)
        rect_x = x - styles["layer_width"] // 2
        rect_y = y - styles["layer_height"] // 2
        
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
        
    def set_layout(self, layout: str) -> None:
        """
        Set the layout direction for the diagram.
        
        Args:
            layout: "horizontal" or "vertical"
        """
        if layout not in ["horizontal", "vertical"]:
            raise ValueError("Layout must be 'horizontal' or 'vertical'")
        self.layout = layout


class GraphvizRenderer:
    """Renders neural network diagrams using Graphviz for high-quality layout."""
    
    def __init__(self):
        """Initialize the Graphviz renderer."""
        pass
        
    def render(self, input_data, theme: Theme) -> str:
        """
        Render graph or layers to SVG string using Graphviz.
        
        Args:
            input_data: NetworkX directed graph or list of layers (backward compatibility)
            theme: Theme object for styling
            
        Returns:
            SVG content as string
        """
        # Handle backward compatibility with list of layers
        if isinstance(input_data, list):
            # For backward compatibility, create a temporary graph from layers
            import networkx as nx
            graph = nx.DiGraph()
            for i, layer in enumerate(input_data):
                layer_name = f"{layer.layer_type}_{i+1}"
                layer.name = layer_name
                graph.add_node(layer_name, layer=layer)
                if i > 0:
                    prev_layer_name = f"{input_data[i-1].layer_type}_{i}"
                    graph.add_edge(prev_layer_name, layer_name)
        else:
            # Input is already a graph
            graph = input_data
        
        if len(graph) == 0:
            return self._empty_svg(theme)
            
        colors = theme.get_colors()
        
        # Create Graphviz directed graph
        dot = graphviz.Digraph(comment='Neural Network')
        dot.attr(rankdir='LR')  # Left to right layout
        dot.attr('graph', bgcolor=colors['background'])
        dot.attr('node', 
                 fontname='Arial',
                 fontsize='10',
                 style='filled',
                 shape='box',
                 margin='0.1')
        dot.attr('edge', 
                 color=colors['connection'],
                 arrowhead='normal')
        
        # Add nodes to the graph
        for node_name in graph.nodes():
            layer = graph.nodes[node_name]['layer']
            label = self._create_html_label(layer, colors)
            dot.node(node_name, label, 
                    fillcolor=self._get_layer_color(layer, colors),
                    fontcolor=colors['text'])
        
        # Add edges to the graph
        for source, target in graph.edges():
            dot.edge(source, target)
        
        # Render to SVG
        svg_content = dot.pipe(format='svg', encoding='utf-8')
        return svg_content
        
    def _empty_svg(self, theme: Theme) -> str:
        """Generate empty SVG for diagrams with no layers."""
        colors = theme.get_colors()
        return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="200" height="100" viewBox="0 0 200 100">
  <rect width="100%" height="100%" fill="{colors['background']}"/>
  <text x="100" y="50" text-anchor="middle" font-family="Arial" font-size="14" fill="{colors['text']}">
    Empty Diagram
  </text>
</svg>'''
        
    def _create_html_label(self, layer: Layer, colors: Dict[str, str]) -> str:
        """Create HTML-like label for a layer node."""
        shape_info = layer.get_shape_info()
        
        # Build table rows for layer information
        header_color = colors.get("layer_stroke", "#333333")
        rows = [
            f'<TR><TD BGCOLOR="{header_color}" COLSPAN="2"><FONT COLOR="white"><B>{layer.layer_type.upper()}</B></FONT></TD></TR>'
        ]
        
        # Add layer-specific information
        if layer.layer_type == 'input':
            rows.append(f'<TR><TD>Shape:</TD><TD>{shape_info.get("shape", "")}</TD></TR>')
        elif layer.layer_type in ['conv', 'conv_transpose']:
            rows.append(f'<TR><TD>Filters:</TD><TD>{shape_info.get("filters", "")}</TD></TR>')
            rows.append(f'<TR><TD>Kernel:</TD><TD>{shape_info.get("kernel_size", "")}</TD></TR>')
            if shape_info.get("stride", 1) != 1:
                rows.append(f'<TR><TD>Stride:</TD><TD>{shape_info.get("stride", "")}</TD></TR>')
            rows.append(f'<TR><TD>Activation:</TD><TD>{shape_info.get("activation", "")}</TD></TR>')
        elif layer.layer_type in ['dense', 'output']:
            rows.append(f'<TR><TD>Units:</TD><TD>{shape_info.get("units", "")}</TD></TR>')
            rows.append(f'<TR><TD>Activation:</TD><TD>{shape_info.get("activation", "")}</TD></TR>')
        elif layer.layer_type == 'dropout':
            rows.append(f'<TR><TD>Rate:</TD><TD>{shape_info.get("rate", "")}</TD></TR>')
        elif layer.layer_type == 'flatten':
            rows.append(f'<TR><TD COLSPAN="2">Flattens input</TD></TR>')
        elif layer.layer_type == 'maxpool':
            rows.append(f'<TR><TD>Pool Size:</TD><TD>{shape_info.get("pool_size", "")}</TD></TR>')
            if shape_info.get("stride") != shape_info.get("pool_size"):
                rows.append(f'<TR><TD>Stride:</TD><TD>{shape_info.get("stride", "")}</TD></TR>')
        elif layer.layer_type == 'upsample':
            rows.append(f'<TR><TD>Size:</TD><TD>{shape_info.get("size", "")}</TD></TR>')
            rows.append(f'<TR><TD>Method:</TD><TD>{shape_info.get("method", "")}</TD></TR>')
        elif layer.layer_type in ['batch_norm', 'layer_norm']:
            rows.append(f'<TR><TD COLSPAN="2">Normalizes activations</TD></TR>')
        elif layer.layer_type == 'multi_head_attention':
            rows.append(f'<TR><TD>Heads:</TD><TD>{shape_info.get("num_heads", "")}</TD></TR>')
            rows.append(f'<TR><TD>Key Dim:</TD><TD>{shape_info.get("key_dim", "")}</TD></TR>')
        elif layer.layer_type == 'embedding':
            rows.append(f'<TR><TD>Vocab Size:</TD><TD>{shape_info.get("vocab_size", "")}</TD></TR>')
            rows.append(f'<TR><TD>Embed Dim:</TD><TD>{shape_info.get("embed_dim", "")}</TD></TR>')
        elif layer.layer_type == 'positional_encoding':
            rows.append(f'<TR><TD>Max Len:</TD><TD>{shape_info.get("max_len", "")}</TD></TR>')
            rows.append(f'<TR><TD>Embed Dim:</TD><TD>{shape_info.get("embed_dim", "")}</TD></TR>')
        elif layer.layer_type == 'reshape':
            rows.append(f'<TR><TD>Shape:</TD><TD>{shape_info.get("target_shape", "")}</TD></TR>')
        elif layer.layer_type == 'global_avg_pool':
            rows.append(f'<TR><TD COLSPAN="2">Global average pooling</TD></TR>')
        elif layer.layer_type == 'concatenate':
            rows.append(f'<TR><TD>Axis:</TD><TD>{shape_info.get("axis", "")}</TD></TR>')
        elif layer.layer_type == 'add':
            rows.append(f'<TR><TD COLSPAN="2">Element-wise addition</TD></TR>')
        
        # Add layer name if it's not auto-generated
        if hasattr(layer, 'name') and not layer.name.startswith(f"{layer.layer_type}_"):
            rows.append(f'<TR><TD COLSPAN="2"><I>{layer.name}</I></TD></TR>')
        
        table_rows = ''.join(rows)
        return f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">{table_rows}</TABLE>>'
        
    def _get_layer_color(self, layer: Layer, colors: Dict[str, str]) -> str:
        """Get appropriate color for a layer type."""
        layer_colors = {
            'input': colors.get('input_fill', colors.get('layer_fill', '#f0f0f0')),
            'conv': colors.get('conv_fill', colors.get('layer_fill', '#f0f0f0')),
            'conv_transpose': colors.get('conv_fill', colors.get('layer_fill', '#f0f0f0')),
            'dense': colors.get('dense_fill', colors.get('layer_fill', '#f0f0f0')),
            'flatten': colors.get('layer_fill', '#f0f0f0'),
            'dropout': colors.get('layer_fill', '#f0f0f0'),
            'output': colors.get('output_fill', colors.get('layer_fill', '#f0f0f0')),
            'maxpool': colors.get('layer_fill', '#e3f2fd'),
            'upsample': colors.get('layer_fill', '#f3e5f5'),
            'batch_norm': colors.get('layer_fill', '#fff3e0'),
            'layer_norm': colors.get('layer_fill', '#fff3e0'),
            'multi_head_attention': colors.get('layer_fill', '#e8f5e8'),
            'embedding': colors.get('layer_fill', '#fce4ec'),
            'positional_encoding': colors.get('layer_fill', '#fce4ec'),
            'reshape': colors.get('layer_fill', '#f9fbe7'),
            'global_avg_pool': colors.get('layer_fill', '#e3f2fd'),
            'concatenate': colors.get('layer_fill', '#ede7f6'),
            'add': colors.get('layer_fill', '#ede7f6')
        }
        return layer_colors.get(layer.layer_type, colors.get('layer_fill', '#f0f0f0'))