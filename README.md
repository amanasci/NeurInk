# NeurInk

**Publication-quality neural network diagram generation library**

NeurInk is a research-grade Python library for creating beautiful, customizable neural network diagrams. It provides both a Python API and a domain-specific language (DSL) for defining network architectures, with SVG rendering and multiple publication themes.

## Features

- 🔬 **Research-grade**: Clean, publication-quality diagrams suitable for papers and presentations
- 🐍 **Python API**: Intuitive method chaining for building network architectures
- 📝 **DSL Support**: Lightweight markup language for defining networks
- 🎨 **Multiple Themes**: IEEE, APJ, Minimal, and Dark themes for different publication styles
- 📐 **SVG Output**: Scalable vector graphics with optional PNG/PDF export
- 🏗️ **Architecture Templates**: Pre-built templates for ResNet, UNet, Transformer, MLP
- 🧪 **Fully Tested**: Comprehensive test suite ensures reliability

## Installation

```bash
pip install neurink
```

For development installation:
```bash
git clone https://github.com/amanasci/NeurInk.git
cd NeurInk
pip install -e .
```

## Quick Start

### Python API

```python
from neurink import Diagram

# Create a neural network diagram using method chaining
diagram = (Diagram()
    .input((64, 64))
    .conv(32, 3)
    .conv(64, 3, stride=2)
    .flatten()
    .dense(128)
    .dropout(0.5)
    .output(10))

# Render to SVG
diagram.render("my_network.svg", theme="ieee")
```

### DSL Usage

```python
from neurink import Diagram

# Define network using DSL
dsl_text = """
input size=28x28
conv filters=32 kernel=3 activation=relu
conv filters=64 kernel=3 activation=relu
flatten
dense units=128 activation=relu
dropout rate=0.5
output units=10 activation=softmax
"""

# Create diagram from DSL
diagram = Diagram.from_string(dsl_text)
diagram.render("dsl_network.svg", theme="minimal")
```

## API Reference

### Core Classes

#### `Diagram`

Main class for building neural network diagrams.

**Methods:**
- `input(shape)` - Add input layer
- `conv(filters, kernel_size, stride=1, activation="relu")` - Add convolutional layer
- `dense(units, activation="relu")` - Add dense layer
- `flatten()` - Add flatten layer
- `dropout(rate)` - Add dropout layer  
- `output(units, activation="softmax")` - Add output layer
- `render(filename, theme="ieee")` - Render to SVG file
- `from_string(dsl_text)` - Create from DSL (class method)

### Themes

Available themes: `"ieee"`, `"apj"`, `"minimal"`, `"dark"`, `"nnsvg"`

Each theme provides different color schemes and styling suitable for various publication formats. The new **`"nnsvg"`** theme provides a beautiful NN-SVG inspired aesthetic with 3D layered blocks, gradients, and shadows.

### Modern Layer Types

NeurInk now supports advanced layer types for complex architectures:

```python
# Transformer components
diagram = (Diagram()
    .input(512)
    .embedding(vocab_size=10000, embed_dim=512)  # Word embeddings
    .layer_norm()                                # Layer normalization
    .attention(num_heads=8, key_dim=64)         # Multi-head attention
    .dense(2048, "gelu")                        # Feed-forward network
    .pooling("global_avg")                      # Global average pooling
    .output(num_classes))

# Modern CNN components  
diagram = (Diagram()
    .input((224, 224, 3))
    .conv(64, 3)
    .batch_norm()                               # Batch normalization
    .pooling("max", pool_size=2, stride=2)     # Max pooling
    .conv(128, 3)
    .batch_norm()
    .pooling("avg", pool_size=2, stride=2)     # Average pooling
    .output(num_classes))
```

### Architecture Templates

Pre-built templates for common architectures with modern components:

```python
from neurink.templates import ResNetTemplate, UNetTemplate, TransformerTemplate, MLPTemplate

# Create ResNet-style architecture with BatchNorm and modern layers
resnet = ResNetTemplate.create(input_shape=(224, 224, 3), num_classes=1000)
resnet.render("resnet.svg", theme="nnsvg")  # Beautiful 3D theme!

# Create modern Transformer with attention layers
transformer = TransformerTemplate.create(vocab_size=10000, max_length=512, num_classes=10)
transformer.render("transformer.svg", theme="nnsvg")

# Create MLP
mlp = MLPTemplate.create(input_size=784, hidden_sizes=[512, 256], num_classes=10)
mlp.render("mlp.svg", theme="nnsvg")
```

## Enhanced DSL Syntax

The NeurInk DSL now supports modern layer types for complex architectures:

```
# Traditional layers
input size=64x64          # Input layer with 64x64 shape
conv filters=32 kernel=3  # Convolutional layer
dense units=128           # Dense layer
flatten                   # Flatten layer (no parameters)
dropout rate=0.5          # Dropout layer

# Modern layers for complex architectures
attention heads=8 key_dim=64     # Multi-head attention
embedding vocab_size=10000 embed_dim=512  # Word embeddings
layernorm                        # Layer normalization
batchnorm                        # Batch normalization
pooling type=max size=2          # Pooling layers
output units=10                  # Output layer
```

### Complete Transformer Example (DSL)
```
input size=512
embedding vocab_size=10000 embed_dim=512
layernorm
attention heads=8 key_dim=64
layernorm
dense units=2048 activation=gelu
dense units=512
pooling type=global_avg
dropout rate=0.1
output units=5
```

See [DSL.md](docs/DSL.md) for complete syntax reference with all layer types.

## Examples

### Basic CNN for Image Classification

```python
from neurink import Diagram

cnn = (Diagram()
    .input((224, 224, 3))
    .conv(32, 3)
    .conv(32, 3)
    .conv(64, 3, stride=2)
    .conv(64, 3)
    .flatten()
    .dense(512)
    .dropout(0.5)
    .output(1000))

cnn.render("cnn_classifier.svg", theme="nnsvg")  # Use the beautiful new theme!
```

### Modern Transformer Architecture

```python
transformer = (Diagram()
    .input(512)
    .embedding(vocab_size=10000, embed_dim=512)
    .layer_norm()
    .attention(num_heads=8, key_dim=64)
    .layer_norm()
    .dense(2048, activation="gelu")
    .dense(512)
    .pooling("global_avg")
    .dropout(0.1)
    .output(num_classes=10))

transformer.render("transformer.svg", theme="nnsvg")
```

### ResNet with Batch Normalization

```python
resnet = (Diagram()
    .input((224, 224, 3))
    .conv(64, 7, stride=2)
    .batch_norm()
    .pooling("max", pool_size=3, stride=2)
    .conv(64, 3)
    .batch_norm()
    .conv(128, 3, stride=2)
    .batch_norm()
    .pooling("global_avg")
    .dense(1000))

resnet.render("resnet.svg", theme="nnsvg")
```

### Simple MLP

```python
mlp = (Diagram()
    .input(784)
    .dense(512)
    .dropout(0.5)
    .dense(256) 
    .dropout(0.5)
    .output(10))

mlp.render("mlp.svg", theme="nnsvg")
```

## Contributing

Contributions are welcome! Please see our development setup:

1. Install development dependencies: `pip install -e .[dev]`
2. Run tests: `pytest`
3. Check code style: `black neurink/` and `flake8 neurink/`
4. Submit pull requests

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use NeurInk in your research, please cite:

```bibtex
@software{neurink,
  title={NeurInk: Publication-quality neural network diagram generation},
  author={NeurInk Contributors},
  url={https://github.com/amanasci/NeurInk},
  year={2024}
}
```

## Changelog

### v0.1.0 (Initial Release)
- Python API with method chaining
- Basic DSL parser
- SVG rendering with 4 themes (IEEE, APJ, Minimal, Dark)
- Architecture templates (ResNet, UNet, Transformer, MLP)  
- Comprehensive test suite
- Complete documentation