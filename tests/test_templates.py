"""
Tests for template system functionality.
"""

import pytest
from neurink.templates import (
    NetworkTemplate, ResNetTemplate, UNetTemplate, 
    TransformerTemplate, MLPTemplate
)
from neurink.diagram import Diagram
from neurink.layer import InputLayer, ConvLayer, DenseLayer, OutputLayer


class TestTemplates:
    """Test cases for template system."""
    
    def test_resnet_template(self):
        """Test ResNet template creation."""
        # Test with default parameters
        resnet = ResNetTemplate.create()
        
        assert isinstance(resnet, Diagram)
        assert len(resnet) > 10  # Should have multiple layers
        
        # Check first layer is input
        assert isinstance(resnet.layers[0], InputLayer)
        assert resnet.layers[0].shape == (224, 224, 3)
        
        # Check last layer is output
        assert isinstance(resnet.layers[-1], OutputLayer)
        assert resnet.layers[-1].units == 1000
        
        # Should have conv layers
        conv_layers = [layer for layer in resnet.layers if isinstance(layer, ConvLayer)]
        assert len(conv_layers) > 5
        
    def test_resnet_template_custom_params(self):
        """Test ResNet template with custom parameters."""
        resnet = ResNetTemplate.create(input_shape=(128, 128, 3), num_classes=10)
        
        # Check custom input shape
        assert resnet.layers[0].shape == (128, 128, 3)
        
        # Check custom output classes
        assert resnet.layers[-1].units == 10
        
    def test_unet_template(self):
        """Test UNet template creation."""
        unet = UNetTemplate.create()
        
        assert isinstance(unet, Diagram)
        assert len(unet) > 10  # Should have multiple layers
        
        # Check first layer is input
        assert isinstance(unet.layers[0], InputLayer)
        assert unet.layers[0].shape == (256, 256, 3)
        
        # Check last layer is output
        assert isinstance(unet.layers[-1], OutputLayer)
        assert unet.layers[-1].units == 1
        assert unet.layers[-1].activation == "sigmoid"
        
        # Should have conv layers
        conv_layers = [layer for layer in unet.layers if isinstance(layer, ConvLayer)]
        assert len(conv_layers) > 8
        
    def test_unet_template_custom_params(self):
        """Test UNet template with custom parameters."""
        unet = UNetTemplate.create(input_shape=(512, 512, 1), num_classes=3)
        
        # Check custom input shape
        assert unet.layers[0].shape == (512, 512, 1)
        
        # Check custom output classes
        assert unet.layers[-1].units == 3
        
    def test_transformer_template(self):
        """Test Transformer template creation."""
        transformer = TransformerTemplate.create()
        
        assert isinstance(transformer, Diagram)
        assert len(transformer) > 5  # Should have multiple layers
        
        # Check first layer is input
        assert isinstance(transformer.layers[0], InputLayer)
        assert transformer.layers[0].shape == (512,)
        
        # Check last layer is output
        assert isinstance(transformer.layers[-1], OutputLayer)
        assert transformer.layers[-1].units == 2
        
        # Should have dense layers (simplified transformer representation)
        dense_layers = [layer for layer in transformer.layers if isinstance(layer, DenseLayer)]
        assert len(dense_layers) > 3
        
    def test_transformer_template_custom_params(self):
        """Test Transformer template with custom parameters."""
        transformer = TransformerTemplate.create(
            vocab_size=50000, max_length=1024, num_classes=5
        )
        
        # Check custom input length
        assert transformer.layers[0].shape == (1024,)
        
        # Check custom output classes
        assert transformer.layers[-1].units == 5
        
    def test_mlp_template(self):
        """Test MLP template creation."""
        mlp = MLPTemplate.create()
        
        assert isinstance(mlp, Diagram)
        assert len(mlp) > 4  # Should have input + hidden + output layers
        
        # Check first layer is input
        assert isinstance(mlp.layers[0], InputLayer)
        assert mlp.layers[0].shape == (784,)
        
        # Check last layer is output
        assert isinstance(mlp.layers[-1], OutputLayer)
        assert mlp.layers[-1].units == 10
        
        # Should have dense layers
        dense_layers = [layer for layer in mlp.layers if isinstance(layer, DenseLayer)]
        assert len(dense_layers) >= 3  # Default hidden sizes [512, 256, 128]
        
    def test_mlp_template_custom_params(self):
        """Test MLP template with custom parameters."""
        mlp = MLPTemplate.create(
            input_size=1000, 
            hidden_sizes=[256, 128, 64], 
            num_classes=5
        )
        
        # Check custom input size
        assert mlp.layers[0].shape == (1000,)
        
        # Check custom output classes
        assert mlp.layers[-1].units == 5
        
        # Should have correct number of dense layers
        dense_layers = [layer for layer in mlp.layers if isinstance(layer, DenseLayer)]
        assert len(dense_layers) == 3  # 3 hidden layers
        
        # Check hidden layer sizes
        assert dense_layers[0].units == 256
        assert dense_layers[1].units == 128
        assert dense_layers[2].units == 64
        
    def test_templates_can_render(self):
        """Test that all templates can be rendered."""
        templates = {
            'resnet': ResNetTemplate.create(),
            'unet': UNetTemplate.create(),
            'transformer': TransformerTemplate.create(),
            'mlp': MLPTemplate.create()
        }
        
        for name, template in templates.items():
            try:
                # Test rendering doesn't crash
                svg_content = template._renderer.render(template.layers, template._get_theme_by_name("ieee"))
                assert isinstance(svg_content, str)
                assert len(svg_content) > 100  # Should be substantial content
                assert '<svg' in svg_content
                assert '</svg>' in svg_content
            except Exception as e:
                pytest.fail(f"Template {name} failed to render: {e}")
                
    def test_template_layer_progression(self):
        """Test that templates have logical layer progressions."""
        # Test ResNet has increasing filter counts
        resnet = ResNetTemplate.create()
        conv_layers = [layer for layer in resnet.layers if isinstance(layer, ConvLayer)]
        
        # Should generally increase in complexity
        filter_counts = [layer.filters for layer in conv_layers]
        assert filter_counts[0] < filter_counts[-1]  # First < Last
        
    def test_template_inheritance(self):
        """Test that templates inherit from NetworkTemplate."""
        templates = [ResNetTemplate, UNetTemplate, TransformerTemplate, MLPTemplate]
        
        for template_class in templates:
            assert issubclass(template_class, NetworkTemplate)
            assert hasattr(template_class, 'create')
            assert callable(template_class.create)
            
    def test_template_documentation(self):
        """Test that templates have proper documentation."""
        templates = [ResNetTemplate, UNetTemplate, TransformerTemplate, MLPTemplate]
        
        for template_class in templates:
            assert template_class.__doc__ is not None
            assert len(template_class.__doc__.strip()) > 0
            
            # Create method should have docs
            assert template_class.create.__doc__ is not None
            assert len(template_class.create.__doc__.strip()) > 0