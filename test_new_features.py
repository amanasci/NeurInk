#!/usr/bin/env python3
"""
Test script for new layer types and enhanced templates.
"""

from neurink import Diagram
from neurink.templates import TransformerTemplate, ResNetTemplate

def test_new_features():
    """Test new layer types and templates."""
    print("Testing new features...")
    
    # Test individual new layer types with NNSVG theme
    print("\n1. Testing individual new layers:")
    simple_transformer = (Diagram()
                         .input(512)
                         .embedding(10000, 512)
                         .layer_norm()
                         .attention(8, 64)
                         .dense(2048)
                         .layer_norm()
                         .output(2))
    
    simple_transformer.render("test_simple_transformer.svg", theme="nnsvg")
    print("   Rendered simple transformer to test_simple_transformer.svg")
    
    # Test CNN with new layers
    print("\n2. Testing CNN with new layers:")
    cnn = (Diagram()
           .input((224, 224, 3))
           .conv(64, 3)
           .batch_norm()
           .pooling("max", 2, 2)
           .conv(128, 3)
           .batch_norm()
           .pooling("avg", 2, 2)
           .dense(256)
           .dropout(0.5)
           .output(1000))
    
    cnn.render("test_enhanced_cnn.svg", theme="nnsvg")
    print("   Rendered enhanced CNN to test_enhanced_cnn.svg")
    
    # Test enhanced templates
    print("\n3. Testing enhanced templates:")
    
    # Enhanced Transformer template
    transformer = TransformerTemplate.create(vocab_size=10000, max_length=512, num_classes=5)
    transformer.render("test_transformer_template.svg", theme="nnsvg")
    print(f"   Enhanced Transformer template: {len(transformer)} layers -> test_transformer_template.svg")
    
    # Enhanced ResNet template
    resnet = ResNetTemplate.create(input_shape=(224, 224, 3), num_classes=1000)
    resnet.render("test_resnet_template.svg", theme="nnsvg")
    print(f"   Enhanced ResNet template: {len(resnet)} layers -> test_resnet_template.svg")
    
    # Test DSL with new layer types
    print("\n4. Testing DSL with new layers:")
    dsl_text = """
    input size=512
    embedding vocab_size=10000 embed_dim=512
    layernorm
    attention heads=8 key_dim=64
    layernorm
    dense units=2048 activation=relu
    dense units=512
    pooling type=global_avg
    dropout rate=0.1
    output units=5
    """
    
    dsl_diagram = Diagram.from_string(dsl_text)
    dsl_diagram.render("test_dsl_new_layers.svg", theme="nnsvg")
    print(f"   DSL diagram: {len(dsl_diagram)} layers -> test_dsl_new_layers.svg")
    
    print("\n=== Testing completed! ===")
    print("All diagrams rendered with the new NN-SVG aesthetic theme.")

if __name__ == "__main__":
    test_new_features()