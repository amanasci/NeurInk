#!/usr/bin/env python3
"""
Advanced Layout Engine Test Script

This script tests the sophisticated Sugiyama-style layout engine
with complex neural network architectures including U-Net and Transformer.
"""

from neurink import Diagram
import sys
import os

def test_basic_functionality():
    """Test basic layout engine functionality."""
    print("=== Testing Basic Functionality ===")
    
    # Simple linear network
    diagram = (Diagram()
        .input((64, 64, 3), name="input")
        .conv(32, 3, name="conv1")
        .batch_norm(name="bn1") 
        .conv(64, 3, name="conv2")
        .output(10, name="output"))
    
    try:
        diagram.render("test_basic.svg", theme="nnsvg")
        print("✓ Basic linear network rendered successfully")
        return True
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        return False

def test_unet_architecture():
    """Test U-Net architecture with skip connections."""
    print("\n=== Testing U-Net Architecture ===")
    
    diagram = Diagram()
    
    # Encoder path
    with diagram.group("Encoder Level 1", style={"fill": "#e3f2fd"}) as enc1:
        enc1.input((256, 256, 3), name="input")
        enc1.conv(64, 3, name="enc1_conv1")
        enc1.conv(64, 3, name="enc1_conv2")
    
    with diagram.group("Encoder Level 2", style={"fill": "#e8f5e8"}) as enc2:
        enc2.pooling("max", 2, name="pool1")
        enc2.conv(128, 3, name="enc2_conv1")
        enc2.conv(128, 3, name="enc2_conv2")
    
    # Bottleneck
    with diagram.group("Bottleneck", style={"fill": "#ffebee"}) as bottleneck:
        bottleneck.pooling("max", 2, name="pool2")
        bottleneck.conv(256, 3, name="bottleneck_conv1")
        bottleneck.conv(256, 3, name="bottleneck_conv2")
    
    # Decoder path
    with diagram.group("Decoder Level 2", style={"fill": "#f3e5f5"}) as dec2:
        dec2.conv(128, 3, name="dec2_conv1")
        dec2.conv(128, 3, name="dec2_conv2")
    
    with diagram.group("Decoder Level 1", style={"fill": "#e1f5fe"}) as dec1:
        dec1.conv(64, 3, name="dec1_conv1")
        dec1.conv(64, 3, name="dec1_conv2")
    
    # Output
    with diagram.group("Output", style={"fill": "#fce4ec"}) as output:
        output.output(1, activation="sigmoid", name="segmentation_mask")
    
    # Add U-Net skip connections
    diagram.add_connection("enc2_conv2", "dec2_conv1", style="skip")
    diagram.add_connection("enc1_conv2", "dec1_conv1", style="skip")
    
    try:
        diagram.render("test_unet.svg", theme="nnsvg")
        print("✓ U-Net architecture rendered successfully")
        print(f"  - {len(diagram.layers)} layers")
        print(f"  - {len(diagram.groups)} groups") 
        print(f"  - {len(diagram.connections)} skip connections")
        return True
    except Exception as e:
        print(f"✗ U-Net test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transformer_architecture():
    """Test Transformer architecture with attention connections."""
    print("\n=== Testing Transformer Architecture ===")
    
    diagram = Diagram()
    
    # Input embedding
    with diagram.group("Input Embedding", style={"fill": "#e8eaf6"}) as embedding:
        embedding.input(512, name="input_tokens")
        embedding.embedding(vocab_size=30000, embed_dim=768, name="word_embed")
        embedding.layer_norm(name="embed_norm")
    
    # Multi-head attention block 1
    with diagram.group("Multi-Head Attention 1", style={"fill": "#f1f8e9"}) as mha1:
        mha1.attention(num_heads=12, key_dim=64, name="mha1")
        mha1.layer_norm(name="mha1_norm")
    
    # Feed-forward network 1
    with diagram.group("Feed-Forward Network 1", style={"fill": "#fff8e1"}) as ffn1:
        ffn1.dense(3072, activation="gelu", name="ffn1_1")
        ffn1.dense(768, name="ffn1_2")
        ffn1.layer_norm(name="ffn1_norm")
    
    # Output classification
    with diagram.group("Output Classification", style={"fill": "#fce4ec"}) as output:
        output.pooling("global_avg", name="pool")
        output.dense(512, name="classifier")
        output.output(1000, name="logits")
    
    # Add residual connections (typical in Transformers)
    diagram.add_connection("embed_norm", "mha1_norm", style="skip")
    diagram.add_connection("mha1_norm", "ffn1_norm", style="skip")
    
    # Add attention connections
    diagram.add_connection("mha1", "classifier", style="attention")
    
    try:
        diagram.render("test_transformer.svg", theme="nnsvg")
        print("✓ Transformer architecture rendered successfully")
        print(f"  - {len(diagram.layers)} layers")
        print(f"  - {len(diagram.groups)} groups")
        print(f"  - {len(diagram.connections)} connections")
        return True
    except Exception as e:
        print(f"✗ Transformer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complex_dag():
    """Test complex DAG with multiple paths and merges."""
    print("\n=== Testing Complex DAG ===")
    
    diagram = Diagram()
    
    # Input
    diagram.input((224, 224, 3), name="input")
    
    # Split into three paths
    with diagram.group("Path A", style={"fill": "#e3f2fd"}) as path_a:
        path_a.conv(32, 7, name="path_a_conv1")
        path_a.conv(64, 3, name="path_a_conv2")
    
    with diagram.group("Path B", style={"fill": "#e8f5e8"}) as path_b:
        path_b.conv(32, 3, name="path_b_conv1")
        path_b.conv(64, 3, name="path_b_conv2")
    
    with diagram.group("Path C", style={"fill": "#fff3e0"}) as path_c:
        path_c.pooling("avg", 4, name="path_c_pool")
        path_c.conv(64, 1, name="path_c_conv")
    
    # Merge paths
    with diagram.group("Fusion", style={"fill": "#f3e5f5"}) as fusion:
        fusion.conv(128, 1, name="fusion_conv")
        fusion.batch_norm(name="fusion_bn")
    
    # Output
    with diagram.group("Output", style={"fill": "#fce4ec"}) as output:
        output.pooling("global_avg", name="final_pool")
        output.output(1000, name="predictions")
    
    # Create multi-path connections
    diagram.add_connection("input", "path_a_conv1", style="skip")
    diagram.add_connection("input", "path_b_conv1", style="skip")
    diagram.add_connection("input", "path_c_pool", style="skip")
    
    # Merge connections
    diagram.add_connection("path_a_conv2", "fusion_conv", style="skip")
    diagram.add_connection("path_b_conv2", "fusion_conv", style="skip")
    diagram.add_connection("path_c_conv", "fusion_conv", style="skip")
    
    try:
        diagram.render("test_complex_dag.svg", theme="nnsvg")
        print("✓ Complex DAG rendered successfully")
        print(f"  - {len(diagram.layers)} layers")
        print(f"  - {len(diagram.groups)} groups")
        print(f"  - {len(diagram.connections)} connections")
        return True
    except Exception as e:
        print(f"✗ Complex DAG test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def visual_inspection_report():
    """Generate a visual inspection report."""
    print("\n=== Visual Inspection Report ===")
    print("Generated test diagrams:")
    print("1. test_basic.svg - Simple linear network")
    print("2. test_unet.svg - U-Net with skip connections") 
    print("3. test_transformer.svg - Transformer with attention")
    print("4. test_complex_dag.svg - Complex multi-path DAG")
    print("\nPlease inspect these SVG files to verify:")
    print("- Layout is hierarchical and readable")
    print("- Skip connections are routed cleanly")
    print("- Groups are visually distinct")
    print("- No overlapping elements")
    print("- Professional appearance")

def main():
    """Run all tests."""
    print("Advanced Layout Engine Test Suite")
    print("==================================")
    
    success_count = 0
    total_tests = 4
    
    if test_basic_functionality():
        success_count += 1
        
    if test_unet_architecture():
        success_count += 1
        
    if test_transformer_architecture():
        success_count += 1
        
    if test_complex_dag():
        success_count += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("✅ All tests passed!")
        visual_inspection_report()
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())