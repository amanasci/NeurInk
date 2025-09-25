#!/usr/bin/env python3
"""
NeurInk v2.0 Example: ResNet-style Architecture

This example demonstrates the new v2.0 features:
- Named layers
- Skip connections using the connect() method
- Graph-based architecture
- Enhanced DSL with connection syntax
"""

from neurink import Diagram

def create_resnet_api():
    """Create ResNet-style architecture using Python API."""
    print("Creating ResNet-style architecture with Python API...")
    
    # Create diagram with named layers
    diagram = Diagram()
    
    # Input layer
    diagram.input((224, 224, 3), name="input")
    
    # Initial convolution
    diagram.conv(64, 7, stride=2, activation="relu", name="conv1")
    
    # First residual block
    diagram.conv(64, 1, activation="relu", name="conv2_1x1")  # 1x1 conv
    diagram.conv(64, 3, activation="relu", name="conv2_3x3")  # 3x3 conv
    diagram.conv(256, 1, activation="linear", name="conv2_1x1_out")  # 1x1 conv (no activation)
    
    # Skip connection from conv1 to conv2_1x1_out
    diagram.connect("conv1", "conv2_1x1_out")
    
    # Second residual block
    diagram.conv(64, 1, activation="relu", name="conv3_1x1")
    diagram.conv(64, 3, activation="relu", name="conv3_3x3")  
    diagram.conv(256, 1, activation="linear", name="conv3_1x1_out")
    
    # Skip connection within block
    diagram.connect("conv2_1x1_out", "conv3_1x1_out")
    
    # Global average pooling and classification
    diagram.flatten(name="global_avg_pool")
    diagram.dense(512, activation="relu", name="fc1")
    diagram.dropout(0.5, name="dropout")
    diagram.output(1000, activation="softmax", name="classification")
    
    print(f"Created {len(diagram)} layers")
    print(f"Layer names: {diagram.get_layer_names()}")
    print(f"Connections (including skip): {len(list(diagram.graph.edges()))}")
    
    # Render the diagram
    diagram.render("resnet_api_example.svg", theme="ieee")
    print("✓ Rendered to resnet_api_example.svg")
    
    return diagram

def create_resnet_dsl():
    """Create ResNet-style architecture using enhanced DSL."""
    print("\nCreating ResNet-style architecture with DSL...")
    
    # Enhanced DSL with named layers and explicit connections
    dsl_text = """
    input size=224x224x3 name=input
    conv filters=64 kernel=7 stride=2 activation=relu name=conv1
    
    conv filters=64 kernel=1 activation=relu name=conv2_1x1
    conv filters=64 kernel=3 activation=relu name=conv2_3x3
    conv filters=256 kernel=1 activation=linear name=conv2_1x1_out
    connect from=conv1 to=conv2_1x1_out
    
    conv filters=64 kernel=1 activation=relu name=conv3_1x1
    conv filters=64 kernel=3 activation=relu name=conv3_3x3
    conv filters=256 kernel=1 activation=linear name=conv3_1x1_out
    connect from=conv2_1x1_out to=conv3_1x1_out
    
    flatten name=global_avg_pool
    dense units=512 activation=relu name=fc1
    dropout rate=0.5 name=dropout
    output units=1000 activation=softmax name=classification
    """
    
    # Parse and create diagram
    diagram = Diagram.from_string(dsl_text)
    
    print(f"Created {len(diagram)} layers")
    print(f"Layer names: {diagram.get_layer_names()}")
    print(f"Connections (including skip): {len(list(diagram.graph.edges()))}")
    
    # Render the diagram
    diagram.render("resnet_dsl_example.svg", theme="ieee")
    print("✓ Rendered to resnet_dsl_example.svg")
    
    return diagram

def compare_approaches(api_diagram, dsl_diagram):
    """Compare the two approaches to ensure they produce equivalent results."""
    print("\nComparing API vs DSL approaches...")
    
    # Check if they have the same structure
    api_edges = set(api_diagram.graph.edges())
    dsl_edges = set(dsl_diagram.graph.edges())
    api_layers = api_diagram.get_layer_names()
    dsl_layers = dsl_diagram.get_layer_names()
    
    print(f"API: {len(api_layers)} layers, {len(api_edges)} edges")
    print(f"DSL: {len(dsl_layers)} layers, {len(dsl_edges)} edges")
    
    layers_match = api_layers == dsl_layers
    edges_match = api_edges == dsl_edges
    
    print(f"Layers identical: {layers_match}")
    print(f"Edges identical: {edges_match}")
    
    if layers_match and edges_match:
        print("🎉 Both approaches produce identical architectures!")
    else:
        print("⚠️  Differences found between approaches")

def simple_skip_connection_example():
    """Simple example showing basic skip connection."""
    print("\nSimple skip connection example...")
    
    diagram = Diagram()
    
    # Create a simple network with skip connection
    diagram.input((28, 28), name="input")
    diagram.conv(32, 3, name="conv1")
    diagram.conv(32, 3, name="conv2")
    diagram.conv(32, 3, name="conv3")
    
    # Add skip connection from conv1 to conv3
    diagram.connect("conv1", "conv3")
    
    diagram.flatten(name="flatten")
    diagram.output(10, name="output")
    
    print("Skip connection: conv1 → conv3")
    print(f"All edges: {list(diagram.graph.edges())}")
    
    diagram.render("simple_skip_example.svg", theme="minimal")
    print("✓ Rendered to simple_skip_example.svg")

def create_unet_api():
    """Create U-Net architecture using Python API."""
    print("\nCreating U-Net architecture with Python API...")
    
    diagram = Diagram()
    
    # Input and encoder path
    diagram.input((256, 256, 1), name="input")
    
    # Encoder (downsampling path)
    diagram.conv(64, 3, activation="relu", name="enc_conv1_1")
    diagram.conv(64, 3, activation="relu", name="enc_conv1_2")
    diagram.maxpool(2, name="pool1")
    
    diagram.conv(128, 3, activation="relu", name="enc_conv2_1") 
    diagram.conv(128, 3, activation="relu", name="enc_conv2_2")
    diagram.maxpool(2, name="pool2")
    
    diagram.conv(256, 3, activation="relu", name="enc_conv3_1")
    diagram.conv(256, 3, activation="relu", name="enc_conv3_2")
    diagram.maxpool(2, name="pool3")
    
    # Bottleneck
    diagram.conv(512, 3, activation="relu", name="bottleneck_conv1")
    diagram.conv(512, 3, activation="relu", name="bottleneck_conv2")
    
    # Decoder (upsampling path)
    diagram.conv_transpose(256, 2, stride=2, activation="relu", name="dec_up3")
    diagram.concatenate(name="concat3")
    diagram.connect("enc_conv3_2", "concat3")  # Skip connection
    diagram.conv(256, 3, activation="relu", name="dec_conv3_1")
    diagram.conv(256, 3, activation="relu", name="dec_conv3_2")
    
    diagram.conv_transpose(128, 2, stride=2, activation="relu", name="dec_up2")
    diagram.concatenate(name="concat2")
    diagram.connect("enc_conv2_2", "concat2")  # Skip connection
    diagram.conv(128, 3, activation="relu", name="dec_conv2_1")
    diagram.conv(128, 3, activation="relu", name="dec_conv2_2")
    
    diagram.conv_transpose(64, 2, stride=2, activation="relu", name="dec_up1")
    diagram.concatenate(name="concat1")
    diagram.connect("enc_conv1_2", "concat1")  # Skip connection
    diagram.conv(64, 3, activation="relu", name="dec_conv1_1")
    diagram.conv(64, 3, activation="relu", name="dec_conv1_2")
    
    # Output layer
    diagram.conv(1, 1, activation="sigmoid", name="output_conv")
    
    print(f"Created {len(diagram)} layers")
    print(f"Skip connections: {len([e for e in diagram.graph.edges() if 'enc' in e[0] and 'concat' in e[1]])}")
    
    diagram.render("unet_api_example.svg", theme="ieee")
    print("✓ Rendered to unet_api_example.svg")
    
    return diagram

def create_unet_dsl():
    """Create U-Net architecture using enhanced DSL."""
    print("\nCreating U-Net architecture with DSL...")
    
    dsl_text = """
    input size=256x256x1 name=input
    
    # Encoder path
    conv filters=64 kernel=3 activation=relu name=enc_conv1_1
    conv filters=64 kernel=3 activation=relu name=enc_conv1_2
    maxpool pool_size=2 name=pool1
    
    conv filters=128 kernel=3 activation=relu name=enc_conv2_1
    conv filters=128 kernel=3 activation=relu name=enc_conv2_2  
    maxpool pool_size=2 name=pool2
    
    conv filters=256 kernel=3 activation=relu name=enc_conv3_1
    conv filters=256 kernel=3 activation=relu name=enc_conv3_2
    maxpool pool_size=2 name=pool3
    
    # Bottleneck
    conv filters=512 kernel=3 activation=relu name=bottleneck_conv1
    conv filters=512 kernel=3 activation=relu name=bottleneck_conv2
    
    # Decoder path
    conv_transpose filters=256 kernel=2 stride=2 activation=relu name=dec_up3
    concatenate name=concat3
    connect from=enc_conv3_2 to=concat3
    conv filters=256 kernel=3 activation=relu name=dec_conv3_1
    conv filters=256 kernel=3 activation=relu name=dec_conv3_2
    
    conv_transpose filters=128 kernel=2 stride=2 activation=relu name=dec_up2
    concatenate name=concat2
    connect from=enc_conv2_2 to=concat2
    conv filters=128 kernel=3 activation=relu name=dec_conv2_1
    conv filters=128 kernel=3 activation=relu name=dec_conv2_2
    
    conv_transpose filters=64 kernel=2 stride=2 activation=relu name=dec_up1
    concatenate name=concat1
    connect from=enc_conv1_2 to=concat1
    conv filters=64 kernel=3 activation=relu name=dec_conv1_1
    conv filters=64 kernel=3 activation=relu name=dec_conv1_2
    
    # Output
    conv filters=1 kernel=1 activation=sigmoid name=output_conv
    """
    
    diagram = Diagram.from_string(dsl_text)
    print(f"Created {len(diagram)} layers")
    print(f"Skip connections: {len([e for e in diagram.graph.edges() if 'enc' in e[0] and 'concat' in e[1]])}")
    
    diagram.render("unet_dsl_example.svg", theme="ieee")
    print("✓ Rendered to unet_dsl_example.svg")
    
    return diagram

def create_transformer_api():
    """Create Transformer encoder architecture using Python API."""
    print("\nCreating Transformer encoder architecture with Python API...")
    
    diagram = Diagram()
    
    # Input processing
    diagram.embedding(10000, 512, name="token_embedding")
    diagram.positional_encoding(5000, 512, name="pos_encoding")
    diagram.add(name="embed_add")
    
    # First transformer block
    diagram.multi_head_attention(8, 64, name="mha1")
    diagram.add(name="skip_add1")
    diagram.connect("embed_add", "skip_add1")  # Residual connection
    diagram.layer_norm(name="norm1")
    
    diagram.dense(2048, activation="relu", name="ffn1_expand")
    diagram.dense(512, activation="linear", name="ffn1_contract")
    diagram.add(name="skip_add2")  
    diagram.connect("norm1", "skip_add2")  # Residual connection
    diagram.layer_norm(name="norm2")
    
    # Second transformer block
    diagram.multi_head_attention(8, 64, name="mha2")
    diagram.add(name="skip_add3")
    diagram.connect("norm2", "skip_add3")  # Residual connection
    diagram.layer_norm(name="norm3")
    
    diagram.dense(2048, activation="relu", name="ffn2_expand")
    diagram.dense(512, activation="linear", name="ffn2_contract")
    diagram.add(name="skip_add4")
    diagram.connect("norm3", "skip_add4")  # Residual connection
    diagram.layer_norm(name="final_norm")
    
    # Output head
    diagram.global_avg_pool(name="pool")
    diagram.dense(256, activation="relu", name="head_dense")
    diagram.output(10, activation="softmax", name="classification")
    
    print(f"Created {len(diagram)} layers")
    print(f"Residual connections: {len([e for e in diagram.graph.edges() if 'skip' in e[1]])}")
    
    diagram.render("transformer_api_example.svg", theme="ieee")
    print("✓ Rendered to transformer_api_example.svg")
    
    return diagram

def create_transformer_dsl():
    """Create Transformer encoder architecture using enhanced DSL."""
    print("\nCreating Transformer encoder architecture with DSL...")
    
    dsl_text = """
    # Input processing
    embedding vocab_size=10000 embed_dim=512 name=token_embedding
    positional_encoding max_len=5000 embed_dim=512 name=pos_encoding
    add name=embed_add
    
    # First transformer block  
    multi_head_attention num_heads=8 key_dim=64 name=mha1
    add name=skip_add1
    connect from=embed_add to=skip_add1
    layer_norm name=norm1
    
    dense units=2048 activation=relu name=ffn1_expand
    dense units=512 activation=linear name=ffn1_contract
    add name=skip_add2
    connect from=norm1 to=skip_add2
    layer_norm name=norm2
    
    # Second transformer block
    multi_head_attention num_heads=8 key_dim=64 name=mha2
    add name=skip_add3
    connect from=norm2 to=skip_add3
    layer_norm name=norm3
    
    dense units=2048 activation=relu name=ffn2_expand
    dense units=512 activation=linear name=ffn2_contract
    add name=skip_add4
    connect from=norm3 to=skip_add4
    layer_norm name=final_norm
    
    # Output head
    global_avg_pool name=pool
    dense units=256 activation=relu name=head_dense
    output units=10 activation=softmax name=classification
    """
    
    diagram = Diagram.from_string(dsl_text)
    print(f"Created {len(diagram)} layers")
    print(f"Residual connections: {len([e for e in diagram.graph.edges() if 'skip' in e[1]])}")
    
    diagram.render("transformer_dsl_example.svg", theme="ieee")
    print("✓ Rendered to transformer_dsl_example.svg")
    
    return diagram

if __name__ == "__main__":
    print("=" * 70)
    print("NeurInk v2.0 - Advanced Architecture Examples")
    print("Demonstrating ResNet, U-Net, and Transformer architectures")
    print("=" * 70)
    
    # Create ResNet using both approaches
    print("\n🏗️ ResNet Examples:")
    api_diagram = create_resnet_api()
    dsl_diagram = create_resnet_dsl()
    compare_approaches(api_diagram, dsl_diagram)
    
    # Create U-Net using both approaches
    print("\n🏥 U-Net Examples:")
    unet_api = create_unet_api()
    unet_dsl = create_unet_dsl()
    compare_approaches(unet_api, unet_dsl)
    
    # Create Transformer using both approaches  
    print("\n🤖 Transformer Examples:")
    transformer_api = create_transformer_api()
    transformer_dsl = create_transformer_dsl()
    compare_approaches(transformer_api, transformer_dsl)
    
    # Show simple skip connection example
    simple_skip_connection_example()
    
    print("\n" + "=" * 70)
    print("Examples complete! Generated files:")
    print("📊 ResNet Examples:")
    print("  - resnet_api_example.svg")
    print("  - resnet_dsl_example.svg") 
    print("🏥 U-Net Examples:")
    print("  - unet_api_example.svg")
    print("  - unet_dsl_example.svg")
    print("🤖 Transformer Examples:")  
    print("  - transformer_api_example.svg")
    print("  - transformer_dsl_example.svg")
    print("🔗 Simple Skip Connection:")
    print("  - simple_skip_example.svg")
    print("=" * 70)