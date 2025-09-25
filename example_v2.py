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

if __name__ == "__main__":
    print("=" * 60)
    print("NeurInk v2.0 - Advanced Architecture Examples")
    print("=" * 60)
    
    # Create ResNet using both approaches
    api_diagram = create_resnet_api()
    dsl_diagram = create_resnet_dsl()
    
    # Compare results
    compare_approaches(api_diagram, dsl_diagram)
    
    # Show simple skip connection example
    simple_skip_connection_example()
    
    print("\n" + "=" * 60)
    print("Examples complete! Generated files:")
    print("- resnet_api_example.svg")
    print("- resnet_dsl_example.svg")
    print("- simple_skip_example.svg")
    print("=" * 60)