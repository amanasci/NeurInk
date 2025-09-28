#!/usr/bin/env python3
"""
NeurInk v2.1 Example: Advanced Visual Annotations and Block Templates

This example demonstrates the new v2.1 features:
- Visual layer annotations (colors, shapes, styles, notes)
- Block templates (@residual, @attention, @encoder)
- Hierarchical block organization
- Enhanced connection types and styling
- Comprehensive styling and documentation features
"""

from neurink import Diagram


def create_visual_annotations_example():
    """Demonstrate comprehensive visual annotation features."""
    print("\n📝 Visual Annotations Example:")
    
    dsl_text = """
    # Visual annotations showcase
    input size=224x224x3 name=input annotation_color=#E3F2FD annotation_note="RGB input images" annotation_shape=hexagon
    
    # Feature extraction with custom colors and shapes
    conv filters=64 kernel=7 stride=2 name=stem_conv annotation_color=#FF6B6B annotation_note="Initial feature extraction" annotation_shape=hexagon highlight=true
    maxpool pool_size=3 stride=2 name=stem_pool annotation_color=#4ECDC4 annotation_shape=diamond
    
    # Processing layers with different styles
    conv filters=128 kernel=3 name=conv1 annotation_color=#45B7D1 annotation_style=bold annotation_note="Feature processing"
    conv filters=256 kernel=3 name=conv2 annotation_color=#96CEB4 annotation_shape=ellipse annotation_style=dashed
    
    # Attention layer highlighted
    multi_head_attention num_heads=8 key_dim=64 name=attention annotation_color=#FCEA2B annotation_shape=circle highlight=true annotation_note="Global attention mechanism"
    
    # Classification head with distinct styling
    flatten name=flatten annotation_style=dotted annotation_note="Flatten for classification"
    dense units=512 name=fc1 annotation_color=#FF9F43 annotation_shape=ellipse annotation_note="Feature dense layer"
    dropout rate=0.5 name=dropout annotation_color=#EE5A52 annotation_style=dashed annotation_note="Regularization"
    output units=1000 name=classifier annotation_color=#00D2D3 annotation_shape=diamond highlight=true annotation_note="Final predictions"
    """
    
    diagram = Diagram.from_string(dsl_text)
    diagram.render("visual_annotations_example.svg", theme="ieee")
    print("✓ Created visual annotations example with custom colors, shapes, and notes")
    return diagram


def create_block_templates_example():
    """Demonstrate block template system."""
    print("\n🧩 Block Templates Example:")
    
    dsl_text = """
    # Advanced architecture using block templates
    input size=224x224x3 name=input annotation_color=#DDD6FE annotation_note="Input layer"
    
    # Stem convolution
    conv filters=64 kernel=7 stride=2 name=stem annotation_color=#F59E0B highlight=true
    
    # ResNet-style blocks with templates
    @residual filters=64 name=res_block1
    @residual filters=128 name=res_block2
    @residual filters=256 name=res_block3
    
    # Global attention
    @attention num_heads=8 key_dim=64 name=global_attention
    
    # Feature encoder with multiple stages
    @encoder filters=[512,1024] use_pooling=true name=feature_encoder
    
    # Final classification
    flatten name=global_pool annotation_shape=circle annotation_color=#8B5CF6
    dense units=512 name=fc annotation_color=#EC4899 annotation_shape=ellipse
    dropout rate=0.3 name=final_dropout annotation_style=dashed
    output units=1000 name=predictions annotation_color=#10B981 annotation_shape=diamond highlight=true
    """
    
    diagram = Diagram.from_string(dsl_text)
    diagram.render("block_templates_example.svg", theme="ieee")
    print("✓ Created block templates example with @residual, @attention, @encoder")
    print(f"  Generated {len(diagram.graph.nodes())} layers from templates")
    return diagram


def create_hierarchical_blocks_example():
    """Demonstrate hierarchical block organization."""
    print("\n🏗️ Hierarchical Blocks Example:")
    
    dsl_text = """
    # Large-scale architecture with hierarchical organization
    input size=224x224x3 name=input annotation_color=#E0E7FF annotation_note="Input images"
    
    backbone {
        # Stem processing
        conv filters=64 kernel=7 stride=2 name=stem_conv annotation_color=#F59E0B annotation_note="Stem convolution"
        maxpool pool_size=3 stride=2 name=stem_pool annotation_color=#F59E0B
        
        # Multi-stage feature extraction
        stage1 {
            @residual filters=64 name=block1_1
            @residual filters=64 name=block1_2
        }
        
        stage2 {
            @residual filters=128 name=block2_1
            @residual filters=128 name=block2_2
            @attention num_heads=4 key_dim=32 name=stage2_attn
        }
        
        stage3 {
            @residual filters=256 name=block3_1
            @residual filters=256 name=block3_2
            @attention num_heads=8 key_dim=64 name=stage3_attn
        }
    }
    
    neck {
        # Feature pyramid network
        conv filters=256 kernel=1 name=fpn_conv1 annotation_color=#EC4899 annotation_note="FPN processing"
        upsample size=2 name=fpn_up1 annotation_color=#EC4899
        conv filters=256 kernel=3 name=fpn_conv2 annotation_color=#EC4899
        
        # Global context
        global_avg_pool name=gap annotation_color=#8B5CF6 annotation_shape=circle
    }
    
    head {
        # Classification head
        dense units=1024 name=head_fc1 annotation_color=#6366F1 annotation_shape=ellipse
        dropout rate=0.5 name=head_dropout annotation_style=dashed
        dense units=512 name=head_fc2 annotation_color=#6366F1 annotation_shape=ellipse  
        output units=1000 name=head_classifier annotation_color=#059669 annotation_shape=diamond highlight=true annotation_note="Final classification"
    }
    """
    
    diagram = Diagram.from_string(dsl_text)
    diagram.render("hierarchical_blocks_example.svg", theme="ieee")
    print("✓ Created hierarchical blocks example with nested organization")
    print(f"  Organized {len(diagram.graph.nodes())} layers into hierarchical structure")
    return diagram


def create_enhanced_connections_example():
    """Demonstrate enhanced connection types and styling."""
    print("\n🔗 Enhanced Connections Example:")
    
    dsl_text = """
    # Showcase different connection types
    input size=128x128x3 name=input annotation_color=#FEF3C7 annotation_note="Multi-path input"
    
    # Parallel processing paths
    conv filters=64 kernel=3 name=path1_conv1 annotation_color=#3B82F6 annotation_note="Path 1 processing"
    conv filters=64 kernel=5 name=path2_conv1 annotation_color=#10B981 annotation_note="Path 2 processing"
    conv filters=64 kernel=7 name=path3_conv1 annotation_color=#F59E0B annotation_note="Path 3 processing"
    
    # Continued processing
    conv filters=128 kernel=3 name=path1_conv2 annotation_color=#3B82F6
    conv filters=128 kernel=3 name=path2_conv2 annotation_color=#10B981
    conv filters=128 kernel=3 name=path3_conv2 annotation_color=#F59E0B
    
    # Fusion layer with multiple connection types
    conv filters=256 kernel=1 name=fusion annotation_color=#EC4899 annotation_shape=diamond annotation_note="Multi-path fusion" highlight=true
    
    # Different connection types and styles
    connect from=path1_conv2 to=fusion type=skip style=dashed label="Path1" weight=0.4
    connect from=path2_conv2 to=fusion type=residual style=bold label="Path2" weight=0.4  
    connect from=path3_conv2 to=fusion type=attention style=dotted label="Path3" weight=0.2
    
    # Add skip connection from input
    connect from=input to=fusion type=skip style=dashed label="Skip" weight=0.1
    
    # Final processing
    flatten name=flatten annotation_style=dotted
    dense units=512 name=fc1 annotation_color=#8B5CF6 annotation_shape=ellipse
    dense units=256 name=fc2 annotation_color=#8B5CF6 annotation_shape=ellipse
    
    # Multi-input final layer
    concatenate name=final_fusion annotation_color=#EF4444 annotation_shape=diamond annotation_note="Final fusion"
    connect from=fc1 to=final_fusion weight=0.7
    connect from=fc2 to=final_fusion weight=0.3
    
    output units=10 name=output annotation_color=#059669 annotation_shape=diamond highlight=true annotation_note="Multi-class output"
    """
    
    diagram = Diagram.from_string(dsl_text)
    diagram.render("enhanced_connections_example.svg", theme="ieee")
    print("✓ Created enhanced connections example with multiple connection types")
    print(f"  Created {len(list(diagram.graph.edges()))} connections between layers")
    return diagram


def create_python_api_example():
    """Demonstrate basic Python API usage (visual annotations via DSL only)."""
    print("\n🐍 Python API Example:")
    
    # Note: Visual annotations are currently only available through DSL
    # The Python API focuses on structural definition
    diagram = Diagram()
    
    # Build architecture using method chaining
    diagram.input((224, 224, 3), name="input")
    diagram.conv(64, 7, stride=2, name="stem")
    diagram.maxpool(3, stride=2, name="stem_pool")
    diagram.conv(128, 3, name="conv1")
    diagram.conv(256, 3, name="conv2")
    diagram.multi_head_attention(8, 64, name="attention")
    diagram.flatten(name="flatten")
    diagram.dense(512, name="fc1")
    diagram.dropout(0.5, name="dropout")
    diagram.output(1000, name="classifier")
    
    # Add skip connection
    diagram.connect("stem", "conv2")
    
    diagram.render("python_api_example.svg", theme="ieee")
    print("✓ Created Python API example (visual annotations available via DSL)")
    print("  Note: For visual styling, use DSL syntax with annotation_* parameters")
    return diagram


def compare_features():
    """Compare v2.0 and v2.1 features."""
    print("\n📊 Feature Comparison:")
    print("v2.0 Features:")
    print("  ✓ Graph-based architecture")
    print("  ✓ Named layers")  
    print("  ✓ Skip connections")
    print("  ✓ Enhanced DSL syntax")
    print("  ✓ Graphviz rendering")
    
    print("\nv2.1 New Features:")
    print("  ✓ Visual layer annotations")
    print("  ✓ Block template system")  
    print("  ✓ Hierarchical blocks")
    print("  ✓ Enhanced connections")
    print("  ✓ Comprehensive styling")


if __name__ == "__main__":
    print("=" * 80)
    print("NeurInk v2.1 - Advanced Visual Features & Block Templates")
    print("Demonstrating visual annotations, block templates, and hierarchical organization")
    print("=" * 80)
    
    # Create comprehensive examples
    visual_diagram = create_visual_annotations_example()
    templates_diagram = create_block_templates_example()
    hierarchical_diagram = create_hierarchical_blocks_example()
    connections_diagram = create_enhanced_connections_example()
    api_diagram = create_python_api_example()
    
    # Feature comparison
    compare_features()
    
    print("\n" + "=" * 80)
    print("Examples complete! Generated files:")
    print("🎨 Visual Annotations:")
    print("  - visual_annotations_example.svg")
    print("🧩 Block Templates:")
    print("  - block_templates_example.svg")
    print("🏗️ Hierarchical Organization:")
    print("  - hierarchical_blocks_example.svg")
    print("🔗 Enhanced Connections:")
    print("  - enhanced_connections_example.svg")
    print("🐍 Python API Example:")
    print("  - python_api_example.svg")
    print("=" * 80)
    
    # Summary statistics
    total_layers = (len(visual_diagram.graph.nodes()) +
                   len(templates_diagram.graph.nodes()) +
                   len(hierarchical_diagram.graph.nodes()) +
                   len(connections_diagram.graph.nodes()) +
                   len(api_diagram.graph.nodes()))
    
    print(f"\n📈 Generated {total_layers} total layers across all examples")
    print("🚀 NeurInk v2.1 showcases complete visual control and modular architecture design!")