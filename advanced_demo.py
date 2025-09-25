#!/usr/bin/env python3
"""
Advanced Layout Features Demo

This script demonstrates the new advanced layout capabilities of NeurInk:
1. Complex custom connections between non-adjacent layers
2. Hierarchical layer grouping with visual bounding boxes
3. Professional Bezier curve routing for skip connections
4. Enhanced visual styling for complex architectures
"""

from neurink import Diagram


def create_advanced_resnet_demo():
    """Create a ResNet-like architecture with proper grouping and skip connections."""
    print("Creating Advanced ResNet with Groups and Custom Connections...")
    
    diagram = Diagram()
    
    # Input processing block
    with diagram.group("Input Processing", style={"fill": "#e8f5e8", "stroke": "#4caf50"}) as input_group:
        input_group.input((224, 224, 3), name="input")
        input_group.conv(64, 7, stride=2, name="conv1")
        input_group.batch_norm(name="bn1")
        input_group.pooling("max", pool_size=3, stride=2, name="maxpool")
    
    # Residual Block 1
    with diagram.group("Residual Block 1", style={"fill": "#fff3e0", "stroke": "#ff9800"}) as res1:
        res1.conv(64, 3, name="res1_conv1")
        res1.batch_norm(name="res1_bn1")
        res1.conv(64, 3, name="res1_conv2")
        res1.batch_norm(name="res1_bn2")
    
    # Residual Block 2  
    with diagram.group("Residual Block 2", style={"fill": "#f3e5f5", "stroke": "#9c27b0"}) as res2:
        res2.conv(128, 3, stride=2, name="res2_conv1")
        res2.batch_norm(name="res2_bn1")
        res2.conv(128, 3, name="res2_conv2")
        res2.batch_norm(name="res2_bn2")
    
    # Residual Block 3
    with diagram.group("Residual Block 3", style={"fill": "#e1f5fe", "stroke": "#03a9f4"}) as res3:
        res3.conv(256, 3, stride=2, name="res3_conv1")
        res3.batch_norm(name="res3_bn1")
        res3.conv(256, 3, name="res3_conv2")
        res3.batch_norm(name="res3_bn2")
    
    # Classification head
    with diagram.group("Classification Head", style={"fill": "#fce4ec", "stroke": "#e91e63"}) as classifier:
        classifier.pooling("global_avg", name="global_pool")
        classifier.dense(512, name="fc1")
        classifier.dropout(0.5, name="dropout")
        classifier.output(1000, name="predictions")
    
    # Add skip connections within residual blocks
    diagram.add_connection("maxpool", "res1_bn2", style="skip")  # Skip connection for res block 1
    diagram.add_connection("res1_bn2", "res2_bn2", style="skip")  # Skip connection for res block 2
    diagram.add_connection("res2_bn2", "res3_bn2", style="skip")  # Skip connection for res block 3
    
    # Add attention-style connections for feature reuse
    diagram.add_connection("res1_conv1", "fc1", style="attention")  # Feature attention
    diagram.add_connection("res2_conv1", "fc1", style="attention")  # Feature attention
    
    diagram.render("advanced_resnet_demo.svg", theme="nnsvg")
    print(f"✓ Generated advanced_resnet_demo.svg ({len(diagram.layers)} layers, {len(diagram.groups)} groups, {len(diagram.connections)} connections)")


def create_transformer_architecture_demo():
    """Create a Transformer architecture with attention connections and layer grouping."""
    print("Creating Advanced Transformer with Attention Connections...")
    
    diagram = Diagram()
    
    # Input embedding block
    with diagram.group("Input Embedding", style={"fill": "#e8eaf6", "stroke": "#3f51b5"}) as embedding:
        embedding.input(512, name="input_tokens")
        embedding.embedding(vocab_size=30000, embed_dim=768, name="word_embed")
        embedding.layer_norm(name="embed_norm")
    
    # Multi-head attention block 1
    with diagram.group("Multi-Head Attention 1", style={"fill": "#f1f8e9", "stroke": "#8bc34a"}) as mha1:
        mha1.attention(num_heads=12, key_dim=64, name="mha1")
        mha1.layer_norm(name="mha1_norm")
    
    # Feed-forward network 1
    with diagram.group("Feed-Forward Network 1", style={"fill": "#fff8e1", "stroke": "#ffc107"}) as ffn1:
        ffn1.dense(3072, activation="gelu", name="ffn1_1")
        ffn1.dense(768, name="ffn1_2")
        ffn1.layer_norm(name="ffn1_norm")
    
    # Multi-head attention block 2
    with diagram.group("Multi-Head Attention 2", style={"fill": "#f9fbe7", "stroke": "#689f38"}) as mha2:
        mha2.attention(num_heads=12, key_dim=64, name="mha2")
        mha2.layer_norm(name="mha2_norm")
    
    # Feed-forward network 2  
    with diagram.group("Feed-Forward Network 2", style={"fill": "#fffde7", "stroke": "#f57f17"}) as ffn2:
        ffn2.dense(3072, activation="gelu", name="ffn2_1")
        ffn2.dense(768, name="ffn2_2")
        ffn2.layer_norm(name="ffn2_norm")
    
    # Output classification
    with diagram.group("Output Classification", style={"fill": "#fce4ec", "stroke": "#c2185b"}) as output:
        output.pooling("global_avg", name="pool")
        output.dense(512, name="classifier")
        output.output(1000, name="logits")
    
    # Add residual connections (typical in Transformers)
    diagram.add_connection("embed_norm", "mha1_norm", style="skip")  # Residual around MHA1
    diagram.add_connection("mha1_norm", "ffn1_norm", style="skip")   # Residual around FFN1
    diagram.add_connection("ffn1_norm", "mha2_norm", style="skip")   # Residual around MHA2
    diagram.add_connection("mha2_norm", "ffn2_norm", style="skip")   # Residual around FFN2
    
    # Add attention connections between attention layers
    diagram.add_connection("mha1", "mha2", style="attention")       # Attention flow
    diagram.add_connection("mha1", "classifier", style="attention")  # Direct attention to output
    diagram.add_connection("mha2", "classifier", style="attention")  # Direct attention to output
    
    diagram.render("advanced_transformer_demo.svg", theme="nnsvg")
    print(f"✓ Generated advanced_transformer_demo.svg ({len(diagram.layers)} layers, {len(diagram.groups)} groups, {len(diagram.connections)} connections)")


def create_unet_architecture_demo():
    """Create a U-Net architecture with encoder-decoder groups and skip connections."""
    print("Creating Advanced U-Net with Encoder-Decoder Structure...")
    
    diagram = Diagram()
    
    # Encoder path
    with diagram.group("Encoder Level 1", style={"fill": "#e3f2fd", "stroke": "#2196f3"}) as enc1:
        enc1.input((256, 256, 3), name="input")
        enc1.conv(64, 3, name="enc1_conv1")
        enc1.conv(64, 3, name="enc1_conv2")
    
    with diagram.group("Encoder Level 2", style={"fill": "#e8f5e8", "stroke": "#4caf50"}) as enc2:
        enc2.pooling("max", 2, name="pool1")
        enc2.conv(128, 3, name="enc2_conv1") 
        enc2.conv(128, 3, name="enc2_conv2")
    
    with diagram.group("Encoder Level 3", style={"fill": "#fff3e0", "stroke": "#ff9800"}) as enc3:
        enc3.pooling("max", 2, name="pool2")
        enc3.conv(256, 3, name="enc3_conv1")
        enc3.conv(256, 3, name="enc3_conv2")
    
    # Bottleneck
    with diagram.group("Bottleneck", style={"fill": "#ffebee", "stroke": "#f44336"}) as bottleneck:
        bottleneck.pooling("max", 2, name="pool3")
        bottleneck.conv(512, 3, name="bottleneck_conv1")
        bottleneck.conv(512, 3, name="bottleneck_conv2")
    
    # Decoder path
    with diagram.group("Decoder Level 3", style={"fill": "#f3e5f5", "stroke": "#9c27b0"}) as dec3:
        dec3.conv(256, 3, name="dec3_conv1")
        dec3.conv(256, 3, name="dec3_conv2")
    
    with diagram.group("Decoder Level 2", style={"fill": "#e1f5fe", "stroke": "#00bcd4"}) as dec2:
        dec2.conv(128, 3, name="dec2_conv1")
        dec2.conv(128, 3, name="dec2_conv2")
    
    with diagram.group("Decoder Level 1", style={"fill": "#e8f5e8", "stroke": "#8bc34a"}) as dec1:
        dec1.conv(64, 3, name="dec1_conv1")
        dec1.conv(64, 3, name="dec1_conv2")
    
    # Output
    with diagram.group("Output", style={"fill": "#fce4ec", "stroke": "#e91e63"}) as output:
        output.output(1, activation="sigmoid", name="segmentation_mask")
    
    # Add U-Net skip connections (encoder to decoder)
    diagram.add_connection("enc3_conv2", "dec3_conv1", style="skip")  # Level 3 skip
    diagram.add_connection("enc2_conv2", "dec2_conv1", style="skip")  # Level 2 skip  
    diagram.add_connection("enc1_conv2", "dec1_conv1", style="skip")  # Level 1 skip
    
    # Add bottleneck connections
    diagram.add_connection("bottleneck_conv2", "dec3_conv1", style="skip")  # Bottleneck to decoder
    
    diagram.render("advanced_unet_demo.svg", theme="nnsvg")
    print(f"✓ Generated advanced_unet_demo.svg ({len(diagram.layers)} layers, {len(diagram.groups)} groups, {len(diagram.connections)} connections)")


def create_complex_multi_path_demo():
    """Create a complex multi-path architecture with various connection types."""
    print("Creating Complex Multi-Path Architecture...")
    
    diagram = Diagram()
    
    # Input processing
    with diagram.group("Input", style={"fill": "#f8f9fa", "stroke": "#6c757d"}) as input_group:
        input_group.input((224, 224, 3), name="input")
    
    # Multi-path processing
    with diagram.group("Path A - Spatial Features", style={"fill": "#e3f2fd", "stroke": "#1976d2"}) as path_a:
        path_a.conv(32, 7, name="spatial_conv1")
        path_a.conv(32, 5, name="spatial_conv2")
        path_a.conv(64, 3, name="spatial_conv3")
    
    with diagram.group("Path B - Detail Features", style={"fill": "#e8f5e8", "stroke": "#388e3c"}) as path_b:
        path_b.conv(32, 3, name="detail_conv1")
        path_b.conv(64, 3, name="detail_conv2")
        path_b.conv(64, 1, name="detail_conv3")
    
    with diagram.group("Path C - Global Context", style={"fill": "#fff3e0", "stroke": "#f57c00"}) as path_c:
        path_c.pooling("avg", 4, name="global_pool")
        path_c.conv(32, 1, name="global_conv1")
        path_c.conv(64, 1, name="global_conv2")
    
    # Feature fusion
    with diagram.group("Feature Fusion", style={"fill": "#f3e5f5", "stroke": "#7b1fa2"}) as fusion:
        fusion.conv(128, 1, name="fusion_conv")
        fusion.batch_norm(name="fusion_bn")
        fusion.attention(num_heads=8, name="fusion_attention")
    
    # Final processing
    with diagram.group("Final Processing", style={"fill": "#fce4ec", "stroke": "#c2185b"}) as final:
        final.pooling("global_avg", name="final_pool")
        final.dense(256, name="final_fc")
        final.output(1000, name="predictions")
    
    # Create multi-path connections
    diagram.add_connection("input", "spatial_conv1", style="skip")
    diagram.add_connection("input", "detail_conv1", style="skip") 
    diagram.add_connection("input", "global_pool", style="skip")
    
    # Cross-path connections
    diagram.add_connection("spatial_conv2", "detail_conv2", style="attention")
    diagram.add_connection("detail_conv2", "global_conv1", style="attention")
    diagram.add_connection("spatial_conv3", "fusion_conv", style="skip")
    diagram.add_connection("detail_conv3", "fusion_conv", style="skip")
    diagram.add_connection("global_conv2", "fusion_conv", style="skip")
    
    # Attention connections to output
    diagram.add_connection("fusion_attention", "final_fc", style="attention")
    diagram.add_connection("spatial_conv3", "final_fc", style="attention")
    
    diagram.render("advanced_multipath_demo.svg", theme="nnsvg")
    print(f"✓ Generated advanced_multipath_demo.svg ({len(diagram.layers)} layers, {len(diagram.groups)} groups, {len(diagram.connections)} connections)")


def main():
    """Generate all advanced architecture demonstrations."""
    print("=== NeurInk Advanced Layout Features Demo ===")
    print("Generating publication-quality diagrams with:")
    print("  • Hierarchical layer grouping with visual bounding boxes")
    print("  • Complex custom connections between non-adjacent layers") 
    print("  • Professional Bezier curve routing for skip connections")
    print("  • Advanced 3D styling and effects")
    print()
    
    # Generate all demo architectures
    create_advanced_resnet_demo()
    create_transformer_architecture_demo()
    create_unet_architecture_demo()
    create_complex_multi_path_demo()
    
    print()
    print("=== Advanced Features Demonstration Complete! ===")
    print("Generated professional-quality diagrams showcasing:")
    print("  ✓ Layer grouping with styled bounding boxes")
    print("  ✓ Skip connections with curved Bezier paths")
    print("  ✓ Attention connections with dotted styling")
    print("  ✓ Multi-path architectures with complex routing")
    print("  ✓ Publication-ready 3D visual effects")
    print()
    print("All diagrams are suitable for research papers and presentations!")


if __name__ == "__main__":
    main()