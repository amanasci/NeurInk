"""
Simple validation test for the refactored system.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neurink import Diagram


def test_basic_architectures():
    """Test basic architectures across themes."""
    print("=== Basic Refactoring Validation ===")
    
    # Test 1: Simple CNN
    print("Creating simple CNN...")
    simple_cnn = Diagram()
    simple_cnn.input((28, 28, 1), name="input")
    simple_cnn.conv(32, 3, name="conv1")
    simple_cnn.pooling("max", name="pool1")
    simple_cnn.conv(64, 3, name="conv2")
    simple_cnn.pooling("max", name="pool2")
    simple_cnn.flatten(name="flatten")
    simple_cnn.dense(128, name="dense1")
    simple_cnn.dropout(0.5, name="dropout")
    simple_cnn.output(10, name="output")
    
    # Test 2: Simple ResNet block
    print("Creating simple ResNet block...")
    resnet_block = Diagram()
    resnet_block.input((64, 64, 64), name="input")
    resnet_block.branch("residual", name="branch")
    resnet_block.conv(64, 1, name="conv1")
    resnet_block.batch_norm(name="bn1")
    resnet_block.conv(64, 3, name="conv2")
    resnet_block.batch_norm(name="bn2")
    resnet_block.conv(64, 1, name="conv3")
    resnet_block.batch_norm(name="bn3")
    resnet_block.merge("add", "residual", name="merge")
    resnet_block.output(1000, name="output")
    
    # Test rendering across all themes
    themes = ["ieee", "apj", "minimal", "dark", "nnsvg"]
    
    for theme in themes:
        print(f"Rendering with {theme} theme...")
        simple_cnn.render(f"simple_cnn_{theme}.svg", theme=theme)
        resnet_block.render(f"resnet_block_{theme}.svg", theme=theme)
        print(f"  ✓ Generated simple_cnn_{theme}.svg")
        print(f"  ✓ Generated resnet_block_{theme}.svg")
    
    print("\n=== Basic Validation Complete! ===")
    print("Generated 10 SVG files for visual inspection")


if __name__ == "__main__":
    test_basic_architectures()