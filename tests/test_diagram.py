"""
Tests for the core Diagram class functionality.
"""

import pytest
import os
import tempfile
from neurink import Diagram
from neurink.layer import InputLayer, ConvLayer, DenseLayer, FlattenLayer, DropoutLayer, OutputLayer


class TestDiagram:
    """Test cases for the Diagram class."""
    
    def test_diagram_creation(self):
        """Test creating an empty diagram."""
        diagram = Diagram()
        assert len(diagram) == 0
        assert diagram.layers == []
        
    def test_input_layer(self):
        """Test adding input layer."""
        diagram = Diagram().input((64, 64))
        assert len(diagram) == 1
        assert isinstance(diagram.layers[0], InputLayer)
        assert diagram.layers[0].shape == (64, 64)
        
    def test_input_layer_single_dimension(self):
        """Test input layer with single dimension."""
        diagram = Diagram().input(784)
        assert len(diagram) == 1
        assert diagram.layers[0].shape == (784,)
        
    def test_conv_layer(self):
        """Test adding convolutional layer."""
        diagram = Diagram().conv(32, 3)
        assert len(diagram) == 1
        assert isinstance(diagram.layers[0], ConvLayer)
        assert diagram.layers[0].filters == 32
        assert diagram.layers[0].kernel_size == (3, 3)
        assert diagram.layers[0].stride == 1
        assert diagram.layers[0].activation == "relu"
        
    def test_conv_layer_with_params(self):
        """Test conv layer with custom parameters."""
        diagram = Diagram().conv(64, (5, 5), stride=2, activation="tanh")
        layer = diagram.layers[0]
        assert layer.filters == 64
        assert layer.kernel_size == (5, 5)
        assert layer.stride == 2
        assert layer.activation == "tanh"
        
    def test_dense_layer(self):
        """Test adding dense layer."""
        diagram = Diagram().dense(128)
        assert len(diagram) == 1
        assert isinstance(diagram.layers[0], DenseLayer)
        assert diagram.layers[0].units == 128
        assert diagram.layers[0].activation == "relu"
        
    def test_flatten_layer(self):
        """Test adding flatten layer."""
        diagram = Diagram().flatten()
        assert len(diagram) == 1
        assert isinstance(diagram.layers[0], FlattenLayer)
        
    def test_dropout_layer(self):
        """Test adding dropout layer."""
        diagram = Diagram().dropout(0.5)
        assert len(diagram) == 1
        assert isinstance(diagram.layers[0], DropoutLayer)
        assert diagram.layers[0].rate == 0.5
        
    def test_dropout_validation(self):
        """Test dropout rate validation."""
        with pytest.raises(ValueError):
            Diagram().dropout(1.5)  # Rate > 1.0
        with pytest.raises(ValueError):
            Diagram().dropout(-0.1)  # Rate < 0.0
            
    def test_output_layer(self):
        """Test adding output layer."""
        diagram = Diagram().output(10)
        assert len(diagram) == 1
        assert isinstance(diagram.layers[0], OutputLayer)
        assert diagram.layers[0].units == 10
        assert diagram.layers[0].activation == "softmax"
        
    def test_method_chaining(self):
        """Test method chaining functionality."""
        diagram = (Diagram()
                  .input((28, 28))
                  .conv(32, 3)
                  .conv(64, 3)
                  .flatten()
                  .dense(128)
                  .dropout(0.5)
                  .output(10))
        
        assert len(diagram) == 7
        assert isinstance(diagram.layers[0], InputLayer)
        assert isinstance(diagram.layers[1], ConvLayer)
        assert isinstance(diagram.layers[2], ConvLayer)
        assert isinstance(diagram.layers[3], FlattenLayer)
        assert isinstance(diagram.layers[4], DenseLayer)
        assert isinstance(diagram.layers[5], DropoutLayer)
        assert isinstance(diagram.layers[6], OutputLayer)
        
    def test_render_svg(self):
        """Test rendering to SVG file."""
        diagram = (Diagram()
                  .input((64, 64))
                  .conv(32, 3)
                  .dense(128)
                  .output(10))
                  
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
            temp_path = f.name
            
        try:
            result_path = diagram.render(temp_path)
            assert result_path == temp_path
            assert os.path.exists(temp_path)
            
            # Check that file contains SVG content
            with open(temp_path, 'r') as f:
                content = f.read()
                assert content.startswith('<?xml version="1.0"')
                assert '<svg' in content
                assert '</svg>' in content
                
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_render_with_different_themes(self):
        """Test rendering with different themes."""
        diagram = Diagram().input((28, 28)).dense(10)
        
        themes = ["ieee", "apj", "minimal", "dark"]
        
        for theme in themes:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
                temp_path = f.name
                
            try:
                diagram.render(temp_path, theme=theme)
                assert os.path.exists(temp_path)
                
                with open(temp_path, 'r') as f:
                    content = f.read()
                    assert '<svg' in content
                    
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
    def test_invalid_theme(self):
        """Test error handling for invalid theme."""
        diagram = Diagram().input((28, 28))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
            temp_path = f.name
            
        try:
            with pytest.raises(ValueError, match="Unknown theme"):
                diagram.render(temp_path, theme="invalid_theme")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    def test_diagram_repr(self):
        """Test string representation of diagram."""
        diagram = Diagram().input((28, 28)).conv(32, 3).dense(10)
        repr_str = repr(diagram)
        assert "Diagram" in repr_str
        assert "input" in repr_str
        assert "conv" in repr_str
        assert "dense" in repr_str