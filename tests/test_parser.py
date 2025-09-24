"""
Tests for the DSL parser functionality.
"""

import pytest
from neurink import Diagram, DSLParseError
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
        dsl = "conv filters=64 kernel=5"
        diagram = Diagram.from_string(dsl)
        
        layer = diagram.layers[0]
        assert layer.filters == 64
        assert layer.kernel_size == (5, 5)
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

    # Error handling tests
    def test_invalid_dsl_type(self):
        """Test error when DSL is not a string."""
        with pytest.raises(DSLParseError, match="DSL text must be a string"):
            Diagram.from_string(123)
            
    def test_unknown_layer_type(self):
        """Test error for unknown layer type."""
        with pytest.raises(DSLParseError, match="Unknown layer type"):
            Diagram.from_string("unknown_layer param=value")
            
    def test_input_missing_size(self):
        """Test error when input layer missing size parameter."""
        with pytest.raises(DSLParseError, match="missing required 'size' parameter"):
            Diagram.from_string("input")
            
    def test_input_empty_size(self):
        """Test error when input size is empty."""
        with pytest.raises(DSLParseError, match="Input size cannot be empty"):
            Diagram.from_string("input size=")
            
    def test_input_negative_dimension(self):
        """Test error when input dimension is negative."""
        with pytest.raises(DSLParseError, match="All dimensions must be positive"):
            Diagram.from_string("input size=-64x64")
            
    def test_input_zero_dimension(self):
        """Test error when input dimension is zero."""
        with pytest.raises(DSLParseError, match="Input dimension must be positive"):
            Diagram.from_string("input size=0")
            
    def test_input_invalid_dimension(self):
        """Test error when input dimension is invalid."""
        with pytest.raises(DSLParseError, match="Invalid dimension value"):
            Diagram.from_string("input size=abc")
            
    def test_conv_missing_filters(self):
        """Test error when conv layer missing filters."""
        with pytest.raises(DSLParseError, match="missing required 'filters' parameter"):
            Diagram.from_string("conv kernel=3")
            
    def test_conv_missing_kernel(self):
        """Test error when conv layer missing kernel."""
        with pytest.raises(DSLParseError, match="missing required 'kernel' parameter"):
            Diagram.from_string("conv filters=32")
            
    def test_conv_negative_filters(self):
        """Test error when conv filters is negative."""
        with pytest.raises(DSLParseError, match="Number of filters must be positive"):
            Diagram.from_string("conv filters=-32 kernel=3")
            
    def test_conv_zero_kernel(self):
        """Test error when conv kernel is zero."""
        with pytest.raises(DSLParseError, match="Kernel size must be positive"):
            Diagram.from_string("conv filters=32 kernel=0")
            
    def test_conv_negative_stride(self):
        """Test error when conv stride is negative."""
        with pytest.raises(DSLParseError, match="Stride must be positive"):
            Diagram.from_string("conv filters=32 kernel=3 stride=-1")
            
    def test_conv_invalid_numeric(self):
        """Test error when conv has invalid numeric parameter."""
        with pytest.raises(DSLParseError, match="Invalid numeric parameter"):
            Diagram.from_string("conv filters=abc kernel=3")
            
    def test_dense_missing_units(self):
        """Test error when dense layer missing units."""
        with pytest.raises(DSLParseError, match="missing required 'units' parameter"):
            Diagram.from_string("dense")
            
    def test_dense_negative_units(self):
        """Test error when dense units is negative."""
        with pytest.raises(DSLParseError, match="Number of units must be positive"):
            Diagram.from_string("dense units=-128")
            
    def test_dense_invalid_units(self):
        """Test error when dense units is invalid."""
        with pytest.raises(DSLParseError, match="Invalid units parameter"):
            Diagram.from_string("dense units=abc")
            
    def test_dropout_missing_rate(self):
        """Test error when dropout missing rate."""
        with pytest.raises(DSLParseError, match="missing required 'rate' parameter"):
            Diagram.from_string("dropout")
            
    def test_dropout_invalid_rate_high(self):
        """Test error when dropout rate too high."""
        with pytest.raises(DSLParseError, match="Dropout rate must be between 0.0 and 1.0"):
            Diagram.from_string("dropout rate=1.5")
            
    def test_dropout_invalid_rate_low(self):
        """Test error when dropout rate too low."""
        with pytest.raises(DSLParseError, match="Dropout rate must be between 0.0 and 1.0"):
            Diagram.from_string("dropout rate=-0.1")
            
    def test_dropout_invalid_rate_format(self):
        """Test error when dropout rate format is invalid."""
        with pytest.raises(DSLParseError, match="Invalid rate parameter"):
            Diagram.from_string("dropout rate=abc")
            
    def test_output_missing_units(self):
        """Test error when output missing units."""
        with pytest.raises(DSLParseError, match="missing required 'units' parameter"):
            Diagram.from_string("output")
            
    def test_output_negative_units(self):
        """Test error when output units is negative."""
        with pytest.raises(DSLParseError, match="Number of output units must be positive"):
            Diagram.from_string("output units=-10")
            
    def test_output_invalid_units(self):
        """Test error when output units is invalid."""
        with pytest.raises(DSLParseError, match="Invalid units parameter"):
            Diagram.from_string("output units=abc")
            
    def test_invalid_parameter_format(self):
        """Test error for invalid parameter format."""
        with pytest.raises(DSLParseError, match="Invalid units parameter"):
            Diagram.from_string("dense units=abc")  # Invalid number
            
    def test_missing_parameter_value(self):
        """Test error for missing parameter in layer definition."""  
        with pytest.raises(DSLParseError, match="missing required 'units' parameter"):
            Diagram.from_string("dense")  # No parameters at all
            
    def test_empty_parameter_key(self):
        """Test error for empty parameter key."""
        with pytest.raises(DSLParseError, match="Invalid parameter format"):
            Diagram.from_string("dense =128")
            
    def test_empty_parameter_value(self):
        """Test error for empty parameter value."""
        with pytest.raises(DSLParseError, match="Invalid parameter format"):
            Diagram.from_string("dense units=")
            
    def test_line_number_in_error(self):
        """Test that error messages include line numbers."""
        dsl = """
        input size=28x28
        invalid_layer
        dense units=128
        """
        
        with pytest.raises(DSLParseError) as exc_info:
            Diagram.from_string(dsl)
        
        # Should mention line 3 where the error occurs
        assert "line 2" in str(exc_info.value)