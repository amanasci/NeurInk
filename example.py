#!/usr/bin/env python3
"""
NeurInk Example Generator

This script generates comprehensive examples showcasing all NeurInk capabilities,
including the new NN-SVG aesthetic, complex architectures with skip connections,
and modern layer types.
"""

from neurink import Diagram
from neurink.templates import ResNetTemplate, UNetTemplate, TransformerTemplate, MLPTemplate


def generate_theme_comparison():
    """Generate theme comparison examples."""
    print("Generating theme comparison examples...")
    
    # Create a network with new layer types for comparison
    network = (Diagram()
               .input((64, 64, 3))
               .conv(32, 3)
               .batch_norm()
               .pooling("max", 2)
               .conv(64, 3)
               .batch_norm()
               .attention(4, 32)
               .layer_norm()
               .dense(128)
               .dropout(0.5)
               .output(10))
    
    themes = ["ieee", "apj", "minimal", "dark", "nnsvg"]
    for theme in themes:
        filename = f"theme_comparison_{theme}.svg"
        network.render(filename, theme=theme)
        print(f"  ✓ Generated {filename}")


def generate_architecture_templates():
    """Generate enhanced architecture template examples."""
    print("\nGenerating architecture template examples...")
    
    templates = {
        "ResNet": ResNetTemplate.create((224, 224, 3), 1000),
        "UNet": UNetTemplate.create((256, 256, 3), 1),
        "Transformer": TransformerTemplate.create(10000, 512, 10),
        "MLP": MLPTemplate.create(784, [512, 256, 128], 10)
    }
    
    for name, template in templates.items():
        filename = f"template_{name.lower()}.svg"
        template.render(filename, theme="nnsvg")
        print(f"  ✓ Generated {filename} ({len(template)} layers)")


def generate_skip_connection_examples():
    """Generate skip connection demonstration examples."""
    print("\nGenerating skip connection examples...")
    
    # ResNet block example
    resnet_block = (Diagram()
                    .input((64, 64, 256))
                    .branch("identity")
                    .conv(64, 1)
                    .batch_norm()
                    .conv(64, 3)
                    .batch_norm()
                    .conv(256, 1)
                    .batch_norm()
                    .merge("add", "identity")
                    .dense(512)
                    .output(1000))
    
    resnet_block.render("example_resnet_block.svg", theme="nnsvg")
    print(f"  ✓ Generated example_resnet_block.svg ({len(resnet_block)} layers)")
    
    # U-Net encoder-decoder with skip connections
    unet_demo = (Diagram()
                 .input((256, 256, 3))
                 .conv(64, 3)
                 .branch("skip1")
                 .pooling("max", 2)
                 .conv(128, 3)
                 .branch("skip2")
                 .pooling("max", 2)
                 .conv(256, 3)
                 .conv(128, 3)
                 .merge("concat", "skip2")
                 .conv(64, 3)
                 .merge("concat", "skip1")
                 .output(1, activation="sigmoid"))
    
    unet_demo.render("example_unet_skip.svg", theme="nnsvg")
    print(f"  ✓ Generated example_unet_skip.svg ({len(unet_demo)} layers)")


def generate_modern_architectures():
    """Generate examples of modern architectures with new layer types."""
    print("\nGenerating modern architecture examples...")
    
    # Vision Transformer inspired
    vision_transformer = (Diagram()
                         .input((224, 224, 3))
                         .conv(16, 16, stride=16)  # Patch embedding
                         .embedding(196, 768)
                         .layer_norm()
                         .attention(12, 64)
                         .layer_norm()
                         .dense(3072, "gelu")
                         .dense(768)
                         .attention(12, 64)
                         .layer_norm()
                         .dense(3072, "gelu")
                         .dense(768)
                         .pooling("global_avg")
                         .layer_norm()
                         .output(1000))
    
    vision_transformer.render("example_vision_transformer.svg", theme="nnsvg")
    print(f"  ✓ Generated example_vision_transformer.svg ({len(vision_transformer)} layers)")
    
    # Modern CNN with advanced layers
    modern_cnn = (Diagram()
                  .input((224, 224, 3))
                  .conv(64, 7, stride=2)
                  .batch_norm()
                  .pooling("max", 3, 2)
                  .conv(128, 3)
                  .batch_norm()
                  .attention(8, 16)  # Attention in CNN
                  .conv(256, 3, stride=2)
                  .batch_norm()
                  .pooling("avg", 2)
                  .conv(512, 3)
                  .batch_norm()
                  .pooling("global_avg")
                  .dense(1024)
                  .layer_norm()
                  .dropout(0.5)
                  .output(1000))
    
    modern_cnn.render("example_modern_cnn.svg", theme="nnsvg")
    print(f"  ✓ Generated example_modern_cnn.svg ({len(modern_cnn)} layers)")


def generate_specialized_examples():
    """Generate specialized architecture examples."""
    print("\nGenerating specialized architecture examples...")
    
    # NLP Transformer
    nlp_transformer = (Diagram()
                      .input(512)
                      .embedding(50000, 768)
                      .layer_norm()
                      .attention(12, 64)
                      .layer_norm()
                      .dense(3072, "gelu")
                      .dense(768)
                      .dropout(0.1)
                      .attention(12, 64)
                      .layer_norm()
                      .dense(3072, "gelu")
                      .dense(768)
                      .dropout(0.1)
                      .pooling("global_avg")
                      .dense(768)
                      .layer_norm()
                      .output(2))
    
    nlp_transformer.render("example_nlp_transformer.svg", theme="nnsvg")
    print(f"  ✓ Generated example_nlp_transformer.svg ({len(nlp_transformer)} layers)")
    
    # Autoencoder
    autoencoder = (Diagram()
                   .input(784)
                   .dense(512)
                   .batch_norm()
                   .dense(256)
                   .batch_norm()
                   .dense(128)  # Bottleneck
                   .dense(256)
                   .batch_norm()
                   .dense(512)
                   .batch_norm()
                   .output(784, activation="sigmoid"))
    
    autoencoder.render("example_autoencoder.svg", theme="nnsvg")
    print(f"  ✓ Generated example_autoencoder.svg ({len(autoencoder)} layers)")


def generate_simple_examples():
    """Generate simple introductory examples."""
    print("\nGenerating simple introductory examples...")
    
    # Simple CNN
    simple_cnn = (Diagram()
                  .input((28, 28, 1))
                  .conv(32, 3)
                  .pooling("max", 2)
                  .conv(64, 3)
                  .pooling("max", 2)
                  .flatten()
                  .dense(128)
                  .dropout(0.5)
                  .output(10))
    
    simple_cnn.render("example_simple_cnn.svg", theme="nnsvg")
    print(f"  ✓ Generated example_simple_cnn.svg ({len(simple_cnn)} layers)")
    
    # Simple MLP
    simple_mlp = (Diagram()
                  .input(784)
                  .dense(512)
                  .dropout(0.3)
                  .dense(256)
                  .dropout(0.3)
                  .dense(128)
                  .output(10))
    
    simple_mlp.render("example_simple_mlp.svg", theme="nnsvg")
    print(f"  ✓ Generated example_simple_mlp.svg ({len(simple_mlp)} layers)")


def main():
    """Generate all examples."""
    print("=== NeurInk Comprehensive Example Generator ===")
    print("Generating all examples with enhanced NN-SVG aesthetic...\n")
    
    # Generate all categories of examples
    generate_theme_comparison()
    generate_architecture_templates()
    generate_skip_connection_examples()
    generate_modern_architectures()
    generate_specialized_examples()
    generate_simple_examples()
    
    print("\n=== Generation Complete! ===")
    print("Generated examples showcasing:")
    print("  • Beautiful NN-SVG inspired 3D aesthetic")
    print("  • Skip connections for ResNet and U-Net")
    print("  • Modern layer types (Attention, LayerNorm, Embedding, etc.)")
    print("  • Comprehensive architecture templates")
    print("  • Theme comparisons across all 5 themes")
    print("  • Simple to complex architecture examples")
    
    print(f"\nTotal examples generated: ~25 SVG files")
    print("All examples use the enhanced NN-SVG theme for publication-quality output.")


if __name__ == "__main__":
    main()