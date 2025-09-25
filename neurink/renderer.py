"""
SVG renderer for neural network diagrams.

Converts layer definitions to clean, scalable SVG output with
theme support and automatic layout.
"""

from typing import List, Dict, Any, Tuple
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
        
        # Check if this is the NNSVG theme to add gradients and shadows
        is_nnsvg = hasattr(theme, 'get_colors') and 'shadow' in colors
        
        defs = f'''  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" 
            refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{colors['connection']}" />
    </marker>'''
        
        if is_nnsvg:
            # Add gradients for 3D effect
            gradient_defs = []
            layer_types = ['input', 'conv', 'dense', 'output', 'flatten', 'dropout', 'attention', 'layernorm', 'embedding', 'pooling', 'batchnorm']
            
            for layer_type in layer_types:
                base_color = colors.get(f'{layer_type}_fill', colors['layer_fill'])
                gradient_defs.append(f'''
    <linearGradient id="grad_{layer_type}" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{base_color};stop-opacity:1" />
      <stop offset="100%" style="stop-color:{self._darken_color(base_color, 0.2)};stop-opacity:1" />
    </linearGradient>''')
            
            # Add shadow filter
            defs += '''
    <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
      <feDropShadow dx="3" dy="3" stdDeviation="2" flood-color="#000" flood-opacity="0.3"/>
    </filter>'''
            
            defs += ''.join(gradient_defs)
        
        defs += '\n  </defs>'
        return defs
        
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
        
        # Check if this is NNSVG theme for 3D effect
        is_nnsvg = 'shadow' in colors and 'layer_depth' in styles
        
        if is_nnsvg:
            # Render with 3D effect
            depth = styles['layer_depth']
            shadow_offset = styles.get('shadow_offset', 4)
            
            # Create 3D layered effect
            layer_svg = f'''  <g class="layer layer-{layer_type}">
    <!-- 3D side faces for depth -->
    <polygon points="{rect_x + styles['layer_width']},{rect_y} {rect_x + styles['layer_width'] + depth},{rect_y - depth} {rect_x + styles['layer_width'] + depth},{rect_y + styles['layer_height'] - depth} {rect_x + styles['layer_width']},{rect_y + styles['layer_height']}"
             fill="{self._darken_color(fill_color, 0.3)}" stroke="{colors['layer_stroke']}" stroke-width="{styles['stroke_width']}" opacity="0.8"/>
    <polygon points="{rect_x},{rect_y + styles['layer_height']} {rect_x + depth},{rect_y + styles['layer_height'] - depth} {rect_x + styles['layer_width'] + depth},{rect_y + styles['layer_height'] - depth} {rect_x + styles['layer_width']},{rect_y + styles['layer_height']}"
             fill="{self._darken_color(fill_color, 0.4)}" stroke="{colors['layer_stroke']}" stroke-width="{styles['stroke_width']}" opacity="0.8"/>
    
    <!-- Main face with gradient -->
    <rect x="{rect_x}" y="{rect_y}" 
          width="{styles['layer_width']}" height="{styles['layer_height']}"
          rx="{styles['border_radius']}" ry="{styles['border_radius']}"
          fill="url(#grad_{layer_type})" stroke="{colors['layer_stroke']}" 
          stroke-width="{styles['stroke_width']}" filter="url(#shadow)"/>
          
    <!-- Text label -->
    <text x="{x}" y="{y + 2}" 
          font-family="{typography['font_family']}" 
          font-size="{typography['font_size']}"
          font-weight="{typography['font_weight']}"
          text-anchor="{typography['text_anchor']}" 
          fill="{colors['text']}">
      {shape_info['display_text']}
    </text>
  </g>'''
        else:
            # Standard flat rendering for other themes
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
        
    def _darken_color(self, hex_color: str, factor: float) -> str:
        """Darken a hex color by a given factor."""
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        
        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16) 
        b = int(hex_color[4:6], 16)
        
        # Darken by factor
        r = max(0, int(r * (1 - factor)))
        g = max(0, int(g * (1 - factor)))
        b = max(0, int(b * (1 - factor)))
        
        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def set_layout(self, layout: str) -> None:
        """
        Set the layout direction for the diagram.
        
        Args:
            layout: "horizontal" or "vertical"
        """
        if layout not in ["horizontal", "vertical"]:
            raise ValueError("Layout must be 'horizontal' or 'vertical'")
        self.layout = layout