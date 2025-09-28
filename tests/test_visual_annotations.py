"""
Tests for visual layer annotations functionality.
"""

import pytest
from neurink import Diagram, DSLParseError
from neurink.parser import DSLParser
from neurink.layer import ConvLayer, DenseLayer


class TestVisualAnnotations:
    """Test cases for visual layer annotations."""
    
    def test_annotation_color_parsing(self):
        """Test parsing custom annotation colors."""
        dsl = "conv filters=32 kernel=3 annotation_color=#FF0000"
        diagram = Diagram.from_string(dsl)
        
        assert len(diagram.graph.nodes) == 1
        layer = list(diagram.graph.nodes(data=True))[0][1]['layer']
        assert layer.annotation_color == "#FF0000"
    
    def test_annotation_shape_parsing(self):
        """Test parsing custom annotation shapes."""
        dsl = "dense units=128 annotation_shape=ellipse"
        diagram = Diagram.from_string(dsl)
        
        assert len(diagram.graph.nodes) == 1
        layer = list(diagram.graph.nodes(data=True))[0][1]['layer']
        assert layer.annotation_shape == "ellipse"
    
    def test_annotation_style_parsing(self):
        """Test parsing custom annotation styles."""
        dsl = "conv filters=64 kernel=5 annotation_style=dashed"
        diagram = Diagram.from_string(dsl)
        
        assert len(diagram.graph.nodes) == 1
        layer = list(diagram.graph.nodes(data=True))[0][1]['layer']
        assert layer.annotation_style == "dashed"
    
    def test_annotation_note_parsing(self):
        """Test parsing annotation notes with quotes."""
        dsl = 'dense units=256 annotation_note="This is a test note"'
        diagram = Diagram.from_string(dsl)
        
        assert len(diagram.graph.nodes) == 1
        layer = list(diagram.graph.nodes(data=True))[0][1]['layer']
        assert layer.annotation_note == "This is a test note"
    
    def test_highlight_parsing(self):
        """Test parsing highlight parameter."""
        test_cases = [
            ("highlight=true", True),
            ("highlight=false", False),
            ("highlight=1", True),
            ("highlight=0", False),
            ("highlight=yes", True),
            ("highlight=no", False),
        ]
        
        for highlight_param, expected in test_cases:
            dsl = f"conv filters=32 kernel=3 {highlight_param}"
            diagram = Diagram.from_string(dsl)
            layer = list(diagram.graph.nodes(data=True))[0][1]['layer']
            assert layer.highlight == expected
    
    def test_multiple_annotations(self):
        """Test multiple annotations on same layer."""
        dsl = '''conv filters=32 kernel=3 annotation_color=#00FF00 annotation_shape=diamond annotation_style=bold annotation_note="Complex layer" highlight=true'''
        diagram = Diagram.from_string(dsl)
        
        assert len(diagram.graph.nodes) == 1
        layer = list(diagram.graph.nodes(data=True))[0][1]['layer']
        
        assert layer.annotation_color == "#00FF00"
        assert layer.annotation_shape == "diamond"
        assert layer.annotation_style == "bold"
        assert layer.annotation_note == "Complex layer"
        assert layer.highlight is True
    
    def test_invalid_annotation_shape(self):
        """Test error handling for invalid annotation shapes."""
        dsl = "conv filters=32 kernel=3 annotation_shape=invalid_shape"
        
        with pytest.raises(DSLParseError) as exc_info:
            Diagram.from_string(dsl)
        assert "Invalid annotation_shape" in str(exc_info.value)
    
    def test_invalid_annotation_style(self):
        """Test error handling for invalid annotation styles."""
        dsl = "dense units=128 annotation_style=invalid_style"
        
        with pytest.raises(DSLParseError) as exc_info:
            Diagram.from_string(dsl)
        assert "Invalid annotation_style" in str(exc_info.value)
    
    def test_invalid_highlight_value(self):
        """Test error handling for invalid highlight values."""
        dsl = "conv filters=32 kernel=3 highlight=maybe"
        
        with pytest.raises(DSLParseError) as exc_info:
            Diagram.from_string(dsl)
        assert "Invalid highlight value" in str(exc_info.value)
    
    def test_rendering_with_annotations(self):
        """Test that diagrams with annotations can be rendered."""
        dsl = '''
        input size=224x224
        conv filters=32 kernel=3 annotation_color=#FF6B6B annotation_note="Feature extractor"
        dense units=128 highlight=true
        output units=10 annotation_shape=ellipse
        '''
        
        diagram = Diagram.from_string(dsl)
        
        # Should not raise an exception
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.svg') as tmp_file:
            result_path = diagram.render(tmp_file.name)
            assert result_path == tmp_file.name
            
            # Check file was created and has content
            with open(result_path, 'r') as f:
                content = f.read()
                assert content.startswith('<?xml')
                assert 'svg' in content.lower()
    
    def test_default_annotation_values(self):
        """Test that layers have default annotation values."""
        dsl = "conv filters=32 kernel=3"
        diagram = Diagram.from_string(dsl)
        
        layer = list(diagram.graph.nodes(data=True))[0][1]['layer']
        
        assert layer.annotation_color is None
        assert layer.annotation_shape == 'box'
        assert layer.annotation_style == 'filled'
        assert layer.annotation_note is None
        assert layer.highlight is False