"""
Test configuration and shared fixtures.
"""

import pytest
import tempfile
import os


@pytest.fixture
def temp_svg_file():
    """Create a temporary SVG file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
        temp_path = f.name
        
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture  
def sample_layers():
    """Create sample layers for testing."""
    from neurink.layer import InputLayer, ConvLayer, DenseLayer, OutputLayer
    
    return [
        InputLayer((28, 28)),
        ConvLayer(32, 3),
        ConvLayer(64, 3),
        DenseLayer(128),
        OutputLayer(10)
    ]