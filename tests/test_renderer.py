"""
Tests for the SVG renderer functionality.
"""

import pytest
import tempfile
import os
from neurink.renderer import SVGRenderer
from neurink.themes import IEEETheme, APJTheme, MinimalTheme, DarkTheme
from neurink.layer import InputLayer, ConvLayer, DenseLayer, OutputLayer


class TestSVGRenderer:
    """Test cases for SVG renderer."""
    
    def test_renderer_creation(self):
        """Test creating an SVG renderer."""
        renderer = SVGRenderer()
        assert renderer is not None
        assert renderer.layout == "horizontal"
        
    def test_empty_diagram_render(self):
        """Test rendering empty diagram."""
        renderer = SVGRenderer()
        theme = IEEETheme()
        
        svg_content = renderer.render([], theme)
        
        assert svg_content.startswith('<?xml version="1.0"')
        assert '<svg' in svg_content
        assert 'Empty Diagram' in svg_content
        assert '</svg>' in svg_content
        
    def test_single_layer_render(self):
        """Test rendering single layer."""
        renderer = SVGRenderer()
        theme = IEEETheme()
        layers = [InputLayer((64, 64))]
        
        svg_content = renderer.render(layers, theme)
        
        assert '<svg' in svg_content
        assert 'Input (64, 64)' in svg_content
        assert '<rect' in svg_content
        assert '<text' in svg_content
        
    def test_multiple_layers_render(self):
        """Test rendering multiple layers."""
        renderer = SVGRenderer()
        theme = IEEETheme()
        layers = [
            InputLayer((28, 28)),
            ConvLayer(32, 3),
            DenseLayer(128),
            OutputLayer(10)
        ]
        
        svg_content = renderer.render(layers, theme)
        
        assert '<svg' in svg_content
        assert 'Input (28, 28)' in svg_content
        assert 'Conv 32@3x3' in svg_content
        assert 'Dense 128' in svg_content
        assert 'Output 10' in svg_content
        assert '<line' in svg_content  # Connection lines
        assert 'marker-end="url(#arrowhead)"' in svg_content
        
    def test_different_themes(self):
        """Test rendering with different themes."""
        renderer = SVGRenderer()
        layers = [InputLayer((64, 64)), DenseLayer(10)]
        
        themes = [IEEETheme(), APJTheme(), MinimalTheme(), DarkTheme()]
        
        for theme in themes:
            svg_content = renderer.render(layers, theme)
            
            assert '<svg' in svg_content
            assert '</svg>' in svg_content
            
            # Check that theme colors are applied
            colors = theme.get_colors()
            assert colors["background"] in svg_content
            
    def test_horizontal_layout(self):
        """Test horizontal layout positioning."""
        renderer = SVGRenderer()
        renderer.set_layout("horizontal")
        theme = IEEETheme()
        layers = [InputLayer((64, 64)), DenseLayer(10)]
        
        styles = theme.get_styles()
        positions = renderer._calculate_positions(layers, styles)
        
        assert len(positions) == 2
        # In horizontal layout, y should be the same, x should increase
        assert positions[0][1] == positions[1][1]  # Same y
        assert positions[0][0] < positions[1][0]   # Increasing x
        
    def test_vertical_layout(self):
        """Test vertical layout positioning."""
        renderer = SVGRenderer()
        renderer.set_layout("vertical")
        theme = IEEETheme()
        layers = [InputLayer((64, 64)), DenseLayer(10)]
        
        styles = theme.get_styles()
        positions = renderer._calculate_positions(layers, styles)
        
        assert len(positions) == 2
        # In vertical layout, x should be the same, y should increase
        assert positions[0][0] == positions[1][0]  # Same x
        assert positions[0][1] < positions[1][1]   # Increasing y
        
    def test_canvas_size_calculation(self):
        """Test canvas size calculation."""
        renderer = SVGRenderer()
        theme = IEEETheme()
        layers = [InputLayer((64, 64)), DenseLayer(10)]
        
        styles = theme.get_styles()
        positions = renderer._calculate_positions(layers, styles)
        width, height = renderer._calculate_canvas_size(positions, styles)
        
        assert width > 0
        assert height > 0
        assert width > styles["layer_width"]
        assert height > styles["layer_height"]
        
    def test_layout_validation(self):
        """Test layout validation."""
        renderer = SVGRenderer()
        
        renderer.set_layout("horizontal")
        assert renderer.layout == "horizontal"
        
        renderer.set_layout("vertical")
        assert renderer.layout == "vertical"
        
        with pytest.raises(ValueError):
            renderer.set_layout("invalid_layout")
            
    def test_layer_specific_colors(self):
        """Test that different layer types get different colors."""
        renderer = SVGRenderer()
        theme = IEEETheme()
        
        input_layer = InputLayer((64, 64))
        conv_layer = ConvLayer(32, 3)
        dense_layer = DenseLayer(128)
        output_layer = OutputLayer(10)
        
        layers = [input_layer, conv_layer, dense_layer, output_layer]
        svg_content = renderer.render(layers, theme)
        
        colors = theme.get_colors()
        
        # Check that different layer-specific colors are used
        assert colors["input_fill"] in svg_content
        assert colors["conv_fill"] in svg_content  
        assert colors["dense_fill"] in svg_content
        assert colors["output_fill"] in svg_content
        
    def test_svg_structure(self):
        """Test SVG structure and elements."""
        renderer = SVGRenderer()
        theme = IEEETheme()
        layers = [InputLayer((64, 64)), DenseLayer(10)]
        
        svg_content = renderer.render(layers, theme)
        
        # Check essential SVG elements
        assert '<?xml version="1.0" encoding="UTF-8"?>' in svg_content
        assert '<svg xmlns="http://www.w3.org/2000/svg"' in svg_content
        assert '<defs>' in svg_content
        assert '<marker id="arrowhead"' in svg_content
        assert '<rect' in svg_content
        assert '<text' in svg_content
        assert '<line' in svg_content
        assert '</svg>' in svg_content