#!/usr/bin/env python3
"""
Example script demonstrating NeurInk library usage.

Shows both Python API and DSL usage for creating neural network diagrams.
"""

from neurink import Diagram

def main():
    print("=== NeurInk Example Usage ===\n")
    
    # Example 1: Python API with method chaining
    print("1. Creating diagram with Python API:")
    diagram1 = (Diagram()
                .input((64, 64))
                .conv(32, 3)
                .conv(64, 3, stride=2)
                .flatten()
                .dense(128)
                .dropout(0.5)
                .output(10))
    
    print(f"   Created diagram with {len(diagram1)} layers")
    print(f"   Layers: {[layer.layer_type for layer in diagram1.layers]}")
    
    # Render to SVG
    output_file = "example_api.svg"
    diagram1.render(output_file, theme="ieee")
    print(f"   Rendered to: {output_file}")
    
    # Example 2: DSL usage
    print("\n2. Creating diagram with DSL:")
    dsl_text = """
    input size=28x28
    conv filters=32 kernel=3 activation=relu
    conv filters=64 kernel=3 activation=relu
    flatten
    dense units=128 activation=relu
    dropout rate=0.5
    output units=10 activation=softmax
    """
    
    diagram2 = Diagram.from_string(dsl_text)
    print(f"   Created diagram with {len(diagram2)} layers")
    print(f"   Layers: {[layer.layer_type for layer in diagram2.layers]}")
    
    # Render with different theme
    output_file2 = "example_dsl.svg"
    diagram2.render(output_file2, theme="minimal")
    print(f"   Rendered to: {output_file2}")
    
    # Example 3: Different themes
    print("\n3. Testing different themes:")
    simple_diagram = Diagram().input((32, 32)).conv(16, 3).dense(10)
    
    themes = ["ieee", "apj", "minimal", "dark"]
    for theme in themes:
        output_file = f"example_{theme}.svg"
        simple_diagram.render(output_file, theme=theme)
        print(f"   Rendered {theme} theme to: {output_file}")
    
    # Example 4: Template usage
    print("\n4. Using architecture templates:")
    from neurink.templates import ResNetTemplate, MLPTemplate
    
    # ResNet template
    resnet = ResNetTemplate.create(input_shape=(224, 224, 3), num_classes=1000)
    resnet.render("example_resnet.svg", theme="ieee")
    print(f"   ResNet template: {len(resnet)} layers -> example_resnet.svg")
    
    # MLP template  
    mlp = MLPTemplate.create(input_size=784, hidden_sizes=[512, 256], num_classes=10)
    mlp.render("example_mlp.svg", theme="minimal")
    print(f"   MLP template: {len(mlp)} layers -> example_mlp.svg")
    
    print("\n=== Examples completed! ===")
    print("Check the generated SVG files to see the diagrams.")

if __name__ == "__main__":
    main()