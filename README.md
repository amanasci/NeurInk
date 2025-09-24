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

Available themes: `"ieee"`, `"apj"`, `"minimal"`, `"dark"`

Each theme provides different color schemes and styling suitable for various publication formats.

### Architecture Templates

Pre-built templates for common architectures:

```python
from neurink.templates import ResNetTemplate, UNetTemplate, TransformerTemplate, MLPTemplate

# Create ResNet-style architecture
resnet = ResNetTemplate.create(input_shape=(224, 224, 3), num_classes=1000)
resnet.render("resnet.svg")

# Create MLP
mlp = MLPTemplate.create(input_size=784, hidden_sizes=[512, 256], num_classes=10)
mlp.render("mlp.svg")
```

## DSL Syntax

The NeurInk DSL provides a simple way to define neural networks:

```
input size=64x64          # Input layer with 64x64 shape
conv filters=32 kernel=3  # Convolutional layer
dense units=128           # Dense layer
flatten                   # Flatten layer (no parameters)
dropout rate=0.5          # Dropout layer
output units=10           # Output layer
```

See [DSL.md](docs/DSL.md) for complete syntax reference.

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

cnn.render("cnn_classifier.svg", theme="ieee")
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

mlp.render("mlp.svg", theme="minimal")
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