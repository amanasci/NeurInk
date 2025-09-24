# NeurInk Markup Language Guide

The NeurInk markup language is a simple, human-readable Domain-Specific Language (DSL) for describing neural network architectures. This guide covers the syntax, supported layers, and advanced features.

## Basic Syntax

Each line represents a layer in your neural network. The basic syntax is:

```
layer_type parameter1=value1 parameter2=value2 ...
```

### Example

```
input size=224x224x3
conv filters=64 kernel=3x3 stride=1 activation=relu
pool kernel=2x2 stride=2
dense units=128 activation=relu
output units=10 activation=softmax
```

## Supported Layers

### Input Layer

Defines the input shape of your network.

```
input size=WIDTH×HEIGHT×CHANNELS
input size=224x224x3      # RGB image
input size=28x28x1        # Grayscale image
input size=1000           # 1D input vector
```

**Parameters:**
- `size`: Input dimensions (required)

### Convolutional Layer (Conv2D)

2D convolutional layer for feature extraction.

```
conv filters=64 kernel=3x3 stride=1 padding=same activation=relu
conv filters=128 kernel=5x5 stride=2 activation=tanh
conv filters=32 kernel=1x1  # 1x1 convolution
```

**Parameters:**
- `filters`: Number of output filters (required)
- `kernel`: Kernel size, format: `WxH` (default: `3x3`)
- `stride`: Stride value (default: `1`)
- `padding`: Padding type (`same`, `valid`) (default: `same`)
- `activation`: Activation function (default: `relu`)

### Pooling Layers

Downsampling layers to reduce spatial dimensions.

```
pool kernel=2x2 stride=2        # Max pooling (default)
maxpool kernel=2x2 stride=2     # Explicit max pooling
avgpool kernel=3x3 stride=1     # Average pooling
```

**Parameters:**
- `kernel`: Pool size, format: `WxH` (default: `2x2`)
- `stride`: Stride value (default: same as kernel)

### Dense Layer (Fully Connected)

Fully connected layer for classification or regression.

```
dense units=128 activation=relu
dense units=64 activation=sigmoid
dense units=10 activation=softmax  # For 10-class classification
```

**Parameters:**
- `units`: Number of neurons (required)
- `activation`: Activation function (default: `relu`)

### Dropout Layer

Regularization layer to prevent overfitting.

```
dropout rate=0.5
dropout rate=0.25
```

**Parameters:**
- `rate`: Dropout rate (0.0 to 1.0) (required)

### Flatten Layer

Flattens multi-dimensional input to 1D.

```
flatten
```

**Parameters:** None

### LSTM Layer

Long Short-Term Memory layer for sequential data.

```
lstm units=128 return_sequences=true
lstm units=64 return_sequences=false
lstm units=256  # return_sequences defaults to false
```

**Parameters:**
- `units`: Number of LSTM units (required)
- `return_sequences`: Whether to return sequences (`true`/`false`) (default: `false`)

### Attention Layer

Self-attention layer for transformer architectures.

```
attention heads=8 dim=64
attention heads=12 dim=128
```

**Parameters:**
- `heads`: Number of attention heads (required)
- `dim`: Dimension per head (required)

### Output Layer

Final layer of the network.

```
output units=10 activation=softmax  # Multi-class classification
output units=1 activation=sigmoid   # Binary classification
output units=1 activation=linear    # Regression
```

**Parameters:**
- `units`: Number of output units (required)
- `activation`: Output activation function (required)

## Activation Functions

Supported activation functions:
- `relu`: Rectified Linear Unit
- `sigmoid`: Sigmoid function
- `tanh`: Hyperbolic tangent
- `softmax`: Softmax (typically for output layers)
- `linear`: Linear/identity function
- `leaky_relu`: Leaky ReLU
- `elu`: Exponential Linear Unit
- `swish`: Swish activation

## Comments

Use `//` for single-line comments:

```
// This is a comment
input size=224x224x3  // Input image size
conv filters=64 kernel=3x3  // First convolutional layer
// pool kernel=2x2  // This line is commented out
dense units=128 activation=relu
```

## Architecture Examples

### Simple CNN for Image Classification

```
input size=224x224x3
conv filters=32 kernel=3x3 activation=relu
pool kernel=2x2
conv filters=64 kernel=3x3 activation=relu
pool kernel=2x2
conv filters=128 kernel=3x3 activation=relu
pool kernel=2x2
flatten
dense units=128 activation=relu
dropout rate=0.5
output units=10 activation=softmax
```

### ResNet-like Architecture

```
input size=224x224x3
conv filters=64 kernel=7x7 stride=2
pool kernel=3x3 stride=2

// ResNet block 1
conv filters=64 kernel=3x3
conv filters=64 kernel=3x3
conv filters=64 kernel=3x3
conv filters=64 kernel=3x3

// ResNet block 2
conv filters=128 kernel=3x3 stride=2
conv filters=128 kernel=3x3
conv filters=128 kernel=3x3
conv filters=128 kernel=3x3

// Classification head
avgpool kernel=7x7
flatten
output units=1000 activation=softmax
```

### LSTM for Sequence Processing

```
input size=100x300  // 100 timesteps, 300 features
lstm units=128 return_sequences=true
dropout rate=0.3
lstm units=128 return_sequences=false
dense units=64 activation=relu
dropout rate=0.5
output units=1 activation=sigmoid
```

### Simple Transformer Block

```
input size=512x768  // 512 tokens, 768 dimensions
attention heads=12 dim=64
dense units=3072 activation=relu  // Feed-forward layer
dense units=768 activation=linear  // Project back
dropout rate=0.1
output units=30522 activation=softmax  // Vocabulary size
```

## Best Practices

### 1. Layer Ordering
- Start with `input` layer
- End with `output` layer
- Use `flatten` before dense layers when transitioning from conv layers

### 2. Parameter Values
- Use powers of 2 for filter numbers (32, 64, 128, 256, 512)
- Common kernel sizes: `3x3`, `5x5`, `7x7`, `1x1`
- Typical dropout rates: `0.2`, `0.3`, `0.5`

### 3. Activation Functions
- Use `relu` for hidden layers
- Use `softmax` for multi-class classification output
- Use `sigmoid` for binary classification output
- Use `linear` for regression output

### 4. Architecture Patterns
- Gradually increase filter sizes in CNN: 32 → 64 → 128 → 256
- Use pooling layers to reduce spatial dimensions
- Add dropout for regularization, especially before output layers

## Error Handling

The parser is forgiving and will:
- Ignore empty lines
- Skip malformed lines with warnings
- Provide default values for missing parameters
- Handle various parameter formats (e.g., `3x3` vs `3×3`)

Common errors and solutions:
- **Missing layer type**: Ensure each line starts with a valid layer type
- **Invalid parameter format**: Use `parameter=value` format
- **Missing required parameters**: Some layers require specific parameters (e.g., `units` for dense layers)

## Future Features

Planned enhancements to the markup language:
- Hierarchical grouping with `{}` syntax
- Variable definitions and reuse
- Conditional layers
- Loop constructs for repetitive patterns
- Import/export of common building blocks