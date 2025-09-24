# NeurInk Tutorial

This tutorial walks you through the complete NeurInk workflow, from installation to creating publication-ready neural network diagrams.

## Table of Contents

1. [Installation](#installation)
2. [Your First Diagram](#your-first-diagram)
3. [Python API Deep Dive](#python-api-deep-dive)
4. [DSL Tutorial](#dsl-tutorial)
5. [Themes and Styling](#themes-and-styling)
6. [Architecture Templates](#architecture-templates)
7. [Advanced Features](#advanced-features)
8. [Best Practices](#best-practices)

## Installation

### Basic Installation

```bash
pip install neurink
```

### Development Installation

For contributing or customizing NeurInk:

```bash
git clone https://github.com/amanasci/NeurInk.git
cd NeurInk
pip install -e .[dev]
```

### Verify Installation

```python
import neurink
print(neurink.__version__)  # Should print version number

# Quick test
from neurink import Diagram
diagram = Diagram().input(28).dense(10)
print(f"Created diagram with {len(diagram)} layers")
```

## Your First Diagram

Let's create a simple neural network diagram:

```python
from neurink import Diagram

# Create a simple feedforward network
my_first_diagram = (Diagram()
    .input(784)           # MNIST flattened input
    .dense(512)           # Hidden layer
    .dense(256)           # Another hidden layer  
    .output(10))          # 10 classes

# Render to SVG
my_first_diagram.render("my_first_network.svg")
```

This creates an SVG file showing a simple 3-layer neural network with clean, professional styling.

## Python API Deep Dive

### Method Chaining

NeurInk uses method chaining for intuitive network construction:

```python
from neurink import Diagram

# Each method returns the diagram object, allowing chaining
diagram = Diagram()
diagram = diagram.input((28, 28))     # Add input layer
diagram = diagram.conv(32, 3)         # Add conv layer
diagram = diagram.dense(128)          # Add dense layer
diagram = diagram.output(10)          # Add output layer

# Or chain in one statement:
diagram = (Diagram()
    .input((28, 28))
    .conv(32, 3)
    .dense(128)
    .output(10))
```

### Layer Types and Parameters

#### Input Layer

```python
# Different input shapes
diagram.input(784)              # 1D input (e.g., flattened MNIST)
diagram.input((28, 28))         # 2D input (e.g., grayscale image)
diagram.input((224, 224, 3))    # 3D input (e.g., RGB image)
```

#### Convolutional Layer

```python
# Basic usage
diagram.conv(32, 3)                    # 32 filters, 3x3 kernel

# With all parameters
diagram.conv(64, (5, 5), stride=2, activation="relu")

# Different activations
diagram.conv(128, 3, activation="tanh")
```

#### Dense (Fully Connected) Layer

```python
# Basic usage
diagram.dense(128)                     # 128 units, relu activation

# Custom activation
diagram.dense(256, activation="tanh")
```

#### Regularization and Utility Layers

```python
# Flatten multi-dimensional input
diagram.flatten()

# Dropout for regularization
diagram.dropout(0.5)        # 50% dropout
diagram.dropout(0.25)       # 25% dropout

# Output layer
diagram.output(10)                        # 10 classes, softmax
diagram.output(1, activation="sigmoid")   # Binary classification
```

### Complete Example: CNN for Image Classification

```python
from neurink import Diagram

# Create a CNN for CIFAR-10 classification
cifar_cnn = (Diagram()
    # Input layer - CIFAR-10 images are 32x32x3
    .input((32, 32, 3))
    
    # First conv block
    .conv(32, 3, stride=1, activation="relu")
    .conv(32, 3, stride=1, activation="relu")
    
    # Second conv block with stride for downsampling
    .conv(64, 3, stride=2, activation="relu")  
    .conv(64, 3, stride=1, activation="relu")
    
    # Third conv block
    .conv(128, 3, stride=2, activation="relu")
    .conv(128, 3, stride=1, activation="relu")
    
    # Classifier head
    .flatten()
    .dense(512, activation="relu")
    .dropout(0.5)
    .dense(256, activation="relu") 
    .dropout(0.3)
    .output(10, activation="softmax")  # 10 CIFAR classes
)

# Render with IEEE theme for publication
cifar_cnn.render("cifar_cnn.svg", theme="ieee")
```

## DSL Tutorial

The Domain-Specific Language provides a text-based way to define networks:

### Basic DSL Usage

```python
from neurink import Diagram

# Define network as text
network_dsl = """
input size=28x28
conv filters=32 kernel=3 activation=relu
conv filters=64 kernel=3 stride=2 activation=relu
flatten
dense units=128 activation=relu
dropout rate=0.5
output units=10 activation=softmax
"""

# Create diagram from DSL
diagram = Diagram.from_string(network_dsl)
diagram.render("dsl_network.svg")
```

### DSL Syntax Rules

1. **One layer per line**: Each line defines one layer
2. **Parameters**: Use `parameter=value` format
3. **Required parameters**: Each layer type has required parameters
4. **Optional parameters**: Have sensible defaults
5. **Whitespace**: Leading/trailing whitespace is ignored

### Layer-by-Layer DSL Guide

#### Input Layer
```
input size=WIDTH_x_HEIGHT           # 2D input
input size=WIDTH_x_HEIGHT_x_CHANNELS # 3D input  
input size=DIMENSION                # 1D input
```

#### Convolutional Layer
```
conv filters=32 kernel=3                    # Required: filters, kernel
conv filters=64 kernel=5 stride=2           # Optional: stride
conv filters=128 kernel=3 activation=tanh   # Optional: activation
```

#### Dense Layer
```
dense units=128                     # Required: units
dense units=256 activation=sigmoid  # Optional: activation
```

#### Other Layers
```
flatten                 # No parameters
dropout rate=0.5       # Required: rate (0.0-1.0)
output units=10        # Required: units, optional: activation
```

### Comparing API vs DSL

Same network, two approaches:

**Python API:**
```python
diagram = (Diagram()
    .input((224, 224, 3))
    .conv(64, 7, stride=2)
    .conv(128, 3)
    .conv(256, 3, stride=2) 
    .flatten()
    .dense(1024)
    .dropout(0.5)
    .output(1000))
```

**DSL:**
```python
dsl_text = """
input size=224x224x3
conv filters=64 kernel=7 stride=2
conv filters=128 kernel=3
conv filters=256 kernel=3 stride=2
flatten
dense units=1024
dropout rate=0.5
output units=1000
"""
diagram = Diagram.from_string(dsl_text)
```

## Themes and Styling

NeurInk provides 4 built-in themes for different publication styles:

### Theme Overview

```python
from neurink import Diagram

# Create a sample network
sample = (Diagram()
    .input((64, 64))
    .conv(32, 3)
    .dense(128)
    .output(10))

# Try different themes
themes = ["ieee", "apj", "minimal", "dark"]

for theme in themes:
    sample.render(f"sample_{theme}.svg", theme=theme)
    print(f"Created sample_{theme}.svg")
```

### Theme Characteristics

#### IEEE Theme (`"ieee"`)
- **Use case**: IEEE conferences and journals
- **Style**: Professional, clear colors, rounded corners
- **Colors**: Light backgrounds, distinct layer colors
- **Typography**: Arial, standard weights

#### APJ Theme (`"apj"`)  
- **Use case**: Astrophysical Journal and similar publications
- **Style**: Conservative, high contrast
- **Colors**: More muted, black borders
- **Typography**: Times serif font

#### Minimal Theme (`"minimal"`)
- **Use case**: Clean presentations, minimalist papers
- **Style**: Very clean, subtle colors
- **Colors**: White backgrounds, light borders
- **Typography**: Light font weights

#### Dark Theme (`"dark"`)
- **Use case**: Presentations, screens, dark mode
- **Style**: Dark backgrounds, bright text
- **Colors**: Dark grays/blues with bright accents
- **Typography**: High contrast text

### Custom Themes

You can create custom themes by extending the Theme base class:

```python
from neurink.themes import Theme

class CustomTheme(Theme):
    def get_colors(self):
        return {
            "background": "#f5f5f5",
            "layer_fill": "#ffffff",
            "layer_stroke": "#333333",
            "input_fill": "#e3f2fd",
            "conv_fill": "#fff3e0", 
            "dense_fill": "#f3e5f5",
            "output_fill": "#e8f5e8",
            "connection": "#666666",
            "text": "#000000"
        }
    
    def get_styles(self):
        return {
            "layer_width": 140,
            "layer_height": 70,
            "layer_spacing_x": 160,
            "layer_spacing_y": 110,
            "border_radius": 10,
            "stroke_width": 2,
            "connection_width": 2,
            "padding": 50
        }
    
    def get_typography(self):
        return {
            "font_family": "Helvetica, sans-serif",
            "font_size": "14px",
            "font_weight": "bold",
            "text_anchor": "middle"
        }

# Use custom theme
diagram = Diagram().input((28, 28)).dense(10)
diagram.render("custom_themed.svg", theme=CustomTheme())
```

## Architecture Templates

NeurInk provides pre-built templates for common architectures:

### ResNet Template

```python
from neurink.templates import ResNetTemplate

# Create ResNet-style architecture
resnet = ResNetTemplate.create(
    input_shape=(224, 224, 3),    # ImageNet input size
    num_classes=1000              # ImageNet classes
)

resnet.render("resnet_template.svg", theme="ieee")
print(f"ResNet has {len(resnet)} layers")
```

### UNet Template  

```python
from neurink.templates import UNetTemplate

# Create UNet for segmentation
unet = UNetTemplate.create(
    input_shape=(256, 256, 3),    # Input image size
    num_classes=1                 # Binary segmentation
)

unet.render("unet_template.svg", theme="minimal")
```

### Transformer Template

```python
from neurink.templates import TransformerTemplate

# Create Transformer architecture
transformer = TransformerTemplate.create(
    vocab_size=10000,       # Vocabulary size
    max_length=512,         # Maximum sequence length
    num_classes=2           # Binary classification
)

transformer.render("transformer_template.svg", theme="dark")
```

### MLP Template

```python
from neurink.templates import MLPTemplate

# Create Multi-Layer Perceptron
mlp = MLPTemplate.create(
    input_size=784,                    # MNIST flattened
    hidden_sizes=[512, 256, 128],      # Hidden layer sizes
    num_classes=10                     # 10 digit classes
)

mlp.render("mlp_template.svg", theme="apj")
```

### Customizing Templates

Templates are just starting points - you can modify them:

```python
from neurink.templates import MLPTemplate

# Start with MLP template
base_mlp = MLPTemplate.create(input_size=784, hidden_sizes=[256, 128])

# Add additional layers
enhanced_mlp = (base_mlp
    .dropout(0.3)                    # Add dropout
    .dense(64, activation="tanh")    # Add another layer
    .output(10))                     # Final output

enhanced_mlp.render("enhanced_mlp.svg")
```

## Advanced Features

### Layout Control

```python
from neurink.renderer import SVGRenderer

# Create custom renderer with vertical layout
renderer = SVGRenderer()
renderer.set_layout("vertical")

# Use with diagram
diagram = Diagram().input((28, 28)).conv(32, 3).dense(10)
# Custom rendering would require renderer integration
```

### Inspecting Diagrams

```python
# Check diagram properties
diagram = (Diagram()
    .input((28, 28))
    .conv(32, 3)
    .dense(128)
    .output(10))

print(f"Number of layers: {len(diagram)}")
print(f"Layer types: {[layer.layer_type for layer in diagram.layers]}")

# Inspect individual layers
for i, layer in enumerate(diagram.layers):
    print(f"Layer {i}: {layer.layer_type}")
    print(f"  Parameters: {layer.params}")
    print(f"  Shape info: {layer.get_shape_info()}")
```

### File Organization

```python
import os

# Create output directory
os.makedirs("diagrams", exist_ok=True)

# Organize by theme
themes = ["ieee", "minimal", "dark"]
base_diagram = Diagram().input((28, 28)).conv(32, 3).dense(10)

for theme in themes:
    base_diagram.render(f"diagrams/network_{theme}.svg", theme=theme)
```

## Best Practices

### 1. Choose Appropriate Themes

- **IEEE/APJ**: For academic papers and conferences
- **Minimal**: For clean presentations or minimalist documents
- **Dark**: For screen presentations or dark-themed materials

### 2. Logical Layer Progression

```python
# Good: Logical flow
good_diagram = (Diagram()
    .input((224, 224, 3))     # Start with input
    .conv(64, 3)              # Feature extraction
    .conv(128, 3)             # Increase complexity
    .flatten()                # Prepare for classification
    .dense(512)               # High-level features
    .dropout(0.5)             # Regularization
    .output(1000))            # Final prediction

# Less ideal: Inconsistent progression
confusing_diagram = (Diagram()
    .input(784)
    .dense(1024)              # Very large first layer
    .conv(32, 3)              # Conv after dense (unusual)
    .output(10))
```

### 3. Meaningful Architecture Choices

```python
# Good: Appropriate for image classification
image_cnn = (Diagram()
    .input((224, 224, 3))
    .conv(32, 3)              # Start with reasonable filter count
    .conv(64, 3)              # Gradually increase
    .conv(128, 3, stride=2)   # Downsample
    .flatten()
    .dense(512)
    .output(1000))

# Good: Appropriate for text classification
text_classifier = (Diagram()
    .input(512)               # Text embedding dimension
    .dense(256)               # Reduce dimensionality
    .dropout(0.5)             # Prevent overfitting
    .dense(128)               # Further reduction
    .output(2))               # Binary classification
```

### 4. Documentation and Comments

```python
# Document your architectures
def create_cifar_cnn():
    """
    Creates a CNN architecture suitable for CIFAR-10 classification.
    
    Returns:
        Diagram: CNN with ~1M parameters
    """
    return (Diagram()
        .input((32, 32, 3))           # CIFAR-10 input
        .conv(32, 3)                  # First conv block
        .conv(32, 3)
        .conv(64, 3, stride=2)        # Downsample
        .conv(64, 3)  
        .conv(128, 3, stride=2)       # Downsample again
        .flatten()
        .dense(512)                   # Classifier
        .dropout(0.5)
        .output(10))                  # 10 CIFAR classes

# Save with descriptive names
cifar_cnn = create_cifar_cnn()
cifar_cnn.render("cifar10_cnn_architecture.svg", theme="ieee")
```

### 5. Batch Processing

```python
# Process multiple architectures
architectures = {
    "simple_mlp": Diagram().input(784).dense(128).output(10),
    "deep_mlp": (Diagram().input(784)
                 .dense(512).dropout(0.5)
                 .dense(256).dropout(0.5)
                 .dense(128).output(10)),
    "cnn": (Diagram().input((28, 28, 1))
            .conv(32, 3).conv(64, 3)
            .flatten().dense(128).output(10))
}

# Render all with consistent theme
for name, diagram in architectures.items():
    diagram.render(f"{name}_architecture.svg", theme="ieee")
    print(f"Generated {name}_architecture.svg ({len(diagram)} layers)")
```

## Troubleshooting

### Common Issues

1. **SVG not rendering properly**: Check file path and permissions
2. **Layer parameters**: Verify required parameters for each layer type
3. **Theme errors**: Use valid theme names: "ieee", "apj", "minimal", "dark"
4. **DSL parsing errors**: Check DSL syntax and parameter formatting

### Getting Help

- Check documentation: [README.md](README.md), [DSL.md](docs/DSL.md)
- Review examples in the repository
- Run tests: `pytest` to verify installation
- Open issues on GitHub for bugs or feature requests

---

This completes the NeurInk tutorial! You now have the knowledge to create publication-quality neural network diagrams for your research and presentations.