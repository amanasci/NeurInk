"""
Tests for the DSL parser functionality.
"""

import pytest
from neurink import Diagram
from neurink.parser import DSLParser
from neurink.layer import InputLayer, ConvLayer, DenseLayer, FlattenLayer, DropoutLayer, OutputLayer


class TestDSLParser:
    """Test cases for DSL parser."""
    
    def test_parser_creation(self):
        """Test creating a DSL parser."""
        parser = DSLParser()
        assert parser is not None
        
    def test_parse_input_layer(self):
        """Test parsing input layer."""
        dsl = "input size=64x64"
        diagram = Diagram.from_string(dsl)
        
        assert len(diagram) == 1
        assert isinstance(diagram.layers[0], InputLayer)
        assert diagram.layers[0].shape == (64, 64)
        
    def test_parse_input_single_dim(self):
        """Test parsing single dimension input."""
        dsl = "input size=784"
        diagram = Diagram.from_string(dsl)
        
        assert len(diagram) == 1
        assert diagram.layers[0].shape == (784,)
        
    def test_parse_conv_layer(self):
        """Test parsing convolutional layer."""
        dsl = "conv filters=32 kernel=3 stride=1 activation=relu"
        diagram = Diagram.from_string(dsl)
        
        assert len(diagram) == 1
        assert isinstance(diagram.layers[0], ConvLayer)
        assert diagram.layers[0].filters == 32
        assert diagram.layers[0].kernel_size == (3, 3)
        assert diagram.layers[0].stride == 1
        assert diagram.layers[0].activation == "relu"
        
    def test_parse_conv_defaults(self):
        """Test parsing conv layer with defaults."""
        dsl = "conv filters=64"
        diagram = Diagram.from_string(dsl)
        
        layer = diagram.layers[0]
        assert layer.filters == 64
        assert layer.kernel_size == (3, 3)  # default
        assert layer.stride == 1  # default
        assert layer.activation == "relu"  # default
        
    def test_parse_dense_layer(self):
        """Test parsing dense layer."""
        dsl = "dense units=128 activation=relu"
        diagram = Diagram.from_string(dsl)
        
        assert len(diagram) == 1
        assert isinstance(diagram.layers[0], DenseLayer)
        assert diagram.layers[0].units == 128
        assert diagram.layers[0].activation == "relu"
        
    def test_parse_flatten_layer(self):
        """Test parsing flatten layer."""
        dsl = "flatten"
        diagram = Diagram.from_string(dsl)
        
        assert len(diagram) == 1
        assert isinstance(diagram.layers[0], FlattenLayer)
        
    def test_parse_dropout_layer(self):
        """Test parsing dropout layer."""
        dsl = "dropout rate=0.5"
        diagram = Diagram.from_string(dsl)
        
        assert len(diagram) == 1
        assert isinstance(diagram.layers[0], DropoutLayer)
        assert diagram.layers[0].rate == 0.5
        
    def test_parse_output_layer(self):
        """Test parsing output layer."""
        dsl = "output units=10 activation=softmax"
        diagram = Diagram.from_string(dsl)
        
        assert len(diagram) == 1
        assert isinstance(diagram.layers[0], OutputLayer)
        assert diagram.layers[0].units == 10
        assert diagram.layers[0].activation == "softmax"
        
    def test_parse_complete_network(self):
        """Test parsing a complete network."""
        dsl = """
        input size=28x28
        conv filters=32 kernel=3
        conv filters=64 kernel=3
        flatten
        dense units=128 activation=relu
        dropout rate=0.5
        output units=10 activation=softmax
        """
        
        diagram = Diagram.from_string(dsl)
        
        assert len(diagram) == 7
        assert isinstance(diagram.layers[0], InputLayer)
        assert isinstance(diagram.layers[1], ConvLayer)
        assert isinstance(diagram.layers[2], ConvLayer)
        assert isinstance(diagram.layers[3], FlattenLayer)
        assert isinstance(diagram.layers[4], DenseLayer)
        assert isinstance(diagram.layers[5], DropoutLayer)
        assert isinstance(diagram.layers[6], OutputLayer)
        
    def test_parse_empty_string(self):
        """Test parsing empty string."""
        dsl = ""
        diagram = Diagram.from_string(dsl)
        assert len(diagram) == 0
        
    def test_parse_whitespace_and_comments(self):
        """Test parsing with extra whitespace."""
        dsl = """
        
        input size=64x64
        
        conv filters=32 kernel=3
        
        dense units=10
        
        """
        
        diagram = Diagram.from_string(dsl)
        assert len(diagram) == 3
        
    def test_parse_mixed_case(self):
        """Test that parser handles parameters correctly."""
        dsl = "conv filters=32 kernel=5 activation=tanh"
        diagram = Diagram.from_string(dsl)
        
        layer = diagram.layers[0]
        assert layer.filters == 32
        assert layer.kernel_size == (5, 5)
        assert layer.activation == "tanh"