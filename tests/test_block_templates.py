"""
Tests for block template functionality.
"""

import pytest
from neurink import Diagram, DSLParseError, get_block_registry
from neurink.blocks import (
    BlockTemplate, ResidualBlockTemplate, AttentionBlockTemplate, 
    EncoderBlockTemplate, BlockTemplateRegistry
)


class TestBlockTemplates:
    """Test cases for block templates."""
    
    def test_block_template_registry(self):
        """Test block template registry functionality."""
        registry = get_block_registry()
        
        # Test that built-in templates are registered
        templates = registry.list_templates()
        assert 'residual' in templates
        assert 'attention' in templates
        assert 'encoder' in templates
    
    def test_residual_block_template_expansion(self):
        """Test residual block template expansion."""
        template = ResidualBlockTemplate()
        expanded = template.expand(filters=64, block_name='res1')
        
        assert 'res1' in expanded
        assert 'conv filters=64' in expanded
        assert 'batch_norm' in expanded
        assert 'connect' in expanded
        assert 'type=skip' in expanded
    
    def test_attention_block_template_expansion(self):
        """Test attention block template expansion."""
        template = AttentionBlockTemplate()
        expanded = template.expand(num_heads=8, key_dim=64, block_name='attn1')
        
        assert 'attn1' in expanded
        assert 'multi_head_attention num_heads=8' in expanded
        assert 'layer_norm' in expanded
        assert 'dense units=512' in expanded  # 8 * 64 = 512
    
    def test_encoder_block_template_expansion(self):
        """Test encoder block template expansion."""
        template = EncoderBlockTemplate()
        expanded = template.expand(filters=[32, 64], block_name='enc1')
        
        assert 'enc1' in expanded
        assert 'conv filters=32' in expanded
        assert 'conv filters=64' in expanded
        assert 'maxpool' in expanded
    
    def test_template_instantiation_via_dsl(self):
        """Test template instantiation through DSL parsing."""
        dsl = '''
        input size=224x224
        @residual filters=64 name=block1
        @attention num_heads=8 key_dim=64 name=attn1
        output units=10
        '''
        
        diagram = Diagram.from_string(dsl)
        
        # Should have input + residual block layers + attention block layers + output
        assert len(diagram.graph.nodes) > 5  # At least input, multiple block layers, output
        
        # Check that block1 layers exist
        node_names = list(diagram.graph.nodes())
        block1_layers = [name for name in node_names if name.startswith('block1_')]
        assert len(block1_layers) > 0
        
        # Check that attn1 layers exist  
        attn1_layers = [name for name in node_names if name.startswith('attn1_')]
        assert len(attn1_layers) > 0
    
    def test_template_with_list_parameters(self):
        """Test template with list parameters."""
        dsl = '''
        input size=224x224
        @encoder filters=[32,64,128] name=feature_extractor
        output units=10
        '''
        
        diagram = Diagram.from_string(dsl)
        node_names = list(diagram.graph.nodes())
        
        # Should have feature_extractor layers
        extractor_layers = [name for name in node_names if name.startswith('feature_extractor_')]
        assert len(extractor_layers) > 0
    
    def test_template_with_boolean_parameters(self):
        """Test template with boolean parameters."""
        dsl = '''
        input size=224x224
        @encoder filters=[64,128] use_pooling=false name=no_pool
        output units=10
        '''
        
        diagram = Diagram.from_string(dsl)
        # Should parse without error
        assert len(diagram.graph.nodes) > 2
    
    def test_unknown_template_error(self):
        """Test error handling for unknown templates."""
        dsl = '''
        input size=224x224
        @unknown_template param=value
        '''
        
        with pytest.raises(DSLParseError) as exc_info:
            Diagram.from_string(dsl)
        assert "Template expansion error" in str(exc_info.value)
        assert "not found" in str(exc_info.value)
    
    def test_template_parameter_parsing(self):
        """Test template parameter parsing edge cases."""
        # Test numeric parameters
        dsl1 = '@residual filters=128 kernel_size=5 name=test1'
        # Test boolean parameters
        dsl2 = '@encoder use_pooling=true name=test2'
        # Test list parameters
        dsl3 = '@encoder filters=[16,32,64] name=test3'
        
        for dsl in [dsl1, dsl2, dsl3]:
            full_dsl = f"input size=224x224\n{dsl}\noutput units=10"
            diagram = Diagram.from_string(full_dsl)
            assert len(diagram.graph.nodes) > 2  # Should parse successfully
    
    def test_template_in_hierarchical_block(self):
        """Test using templates inside hierarchical blocks."""
        dsl = '''
        input size=224x224
        
        backbone {
            @residual filters=64 name=block1
            @residual filters=128 name=block2
        }
        
        output units=10
        '''
        
        diagram = Diagram.from_string(dsl)
        node_names = list(diagram.graph.nodes())
        
        # Should have backbone_block1_* and backbone_block2_* layers
        backbone_layers = [name for name in node_names if name.startswith('backbone_')]
        assert len(backbone_layers) > 0
    
    def test_custom_template_registration(self):
        """Test registering custom templates."""
        class CustomTemplate(BlockTemplate):
            def __init__(self):
                super().__init__("custom")
            
            def expand(self, **params):
                units = params.get('units', 100)
                return f"dense units={units} name=custom_dense"
            
            def get_required_params(self):
                return []
                
            def get_optional_params(self):
                return {'units': 100}
        
        registry = BlockTemplateRegistry()
        custom_template = CustomTemplate()
        registry.register(custom_template)
        
        # Test template is registered
        assert 'custom' in registry.list_templates()
        
        # Test template expansion
        expanded = registry.expand_template('custom', units=256)
        assert 'dense units=256' in expanded
    
    def test_template_rendering(self):
        """Test that diagrams with templates can be rendered."""
        dsl = '''
        input size=224x224
        @residual filters=64 name=res_block
        @attention num_heads=4 key_dim=32 name=attn_block
        output units=10
        '''
        
        diagram = Diagram.from_string(dsl)
        
        # Should render without errors
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.svg') as tmp_file:
            result_path = diagram.render(tmp_file.name)
            assert result_path == tmp_file.name
            
            # Check file content
            with open(result_path, 'r') as f:
                content = f.read()
                assert content.startswith('<?xml')
                assert 'svg' in content.lower()
    
    def test_template_connections_are_created(self):
        """Test that templates create proper connections."""
        dsl = '''
        input size=224x224
        @residual filters=32 name=simple_res
        output units=10
        '''
        
        diagram = Diagram.from_string(dsl)
        
        # Check that connections exist between template-generated layers
        edges = list(diagram.graph.edges(data=True))
        assert len(edges) > 0  # Should have connections
        
        # Check for skip connections (residual template creates them)
        skip_connections = [edge for edge in edges if edge[2].get('type') == 'skip']
        assert len(skip_connections) > 0  # Should have skip connections from residual template