# NeurInk DSL Reference

The NeurInk Domain-Specific Language (DSL) provides a simple, readable way to define neural network architectures. This document covers the complete DSL syntax, examples, and best practices.

## Basic Syntax

The DSL uses a simple line-based format where each line defines a layer:

```
layer_type parameter1=value1 parameter2=value2
```

### Comments and Whitespace

- Empty lines are ignored
- Leading and trailing whitespace is ignored
- Comments are not yet supported (planned for future versions)

## Layer Types

### Input Layer

Defines the input to the network.

```
input size=HEIGHT_x_WIDTH
input size=HEIGHT_x_WIDTH_x_CHANNELS  
input size=SINGLE_DIMENSION
```

**Examples:**
```
input size=28x28        # 2D input (grayscale image)
input size=224x224x3    # 3D input (RGB image)
input size=784          # 1D input (flattened vector)
```

### Convolutional Layer

Defines a 2D convolutional layer.

```
conv filters=N kernel=K [stride=S] [activation=ACT]
```

**Parameters:**
- `filters` (required): Number of output channels/filters
- `kernel` (required): Kernel size (assumes square kernels)
- `stride` (optional): Stride, default=1
- `activation` (optional): Activation function, default="relu"

**Examples:**
```
conv filters=32 kernel=3                    # 32 3x3 filters, stride=1, relu
conv filters=64 kernel=5 stride=2           # 64 5x5 filters, stride=2, relu
conv filters=128 kernel=3 activation=tanh   # 128 3x3 filters, tanh activation
```

### Dense Layer

Defines a fully connected layer.

```
dense units=N [activation=ACT]
```

**Parameters:**
- `units` (required): Number of output units
- `activation` (optional): Activation function, default="relu"

**Examples:**
```
dense units=128                 # 128 units with relu
dense units=256 activation=tanh # 256 units with tanh
dense units=10 activation=linear # 10 units with linear activation
```

### Flatten Layer

Converts multi-dimensional input to 1D vector.

```
flatten
```

No parameters required.

### Dropout Layer

Applies dropout regularization.

```
dropout rate=R
```

**Parameters:**
- `rate` (required): Dropout rate between 0.0 and 1.0

**Examples:**
```
dropout rate=0.5    # 50% dropout
dropout rate=0.25   # 25% dropout
```

### Output Layer

Defines the final output layer.

```
output units=N [activation=ACT]
```

**Parameters:**
- `units` (required): Number of output units
- `activation` (optional): Activation function, default="softmax"

**Examples:**
```
output units=10                      # 10-class classification
output units=1 activation=sigmoid    # Binary classification
output units=100 activation=linear   # Regression
```

### Attention Layer

Defines a multi-head attention layer for Transformers.

```
attention [heads=H] [key_dim=K]
```

**Parameters:**
- `heads` (optional): Number of attention heads, default=8
- `key_dim` (optional): Dimension of keys/queries, default=64

**Examples:**
```
attention                    # 8 heads, 64 key_dim
attention heads=12           # 12 heads, 64 key_dim
attention heads=16 key_dim=128   # 16 heads, 128 key_dim
```

### Layer Normalization

Applies layer normalization.

```
layernorm
```

No parameters required.

### Embedding Layer

Defines an embedding layer for sequence models.

```
embedding vocab_size=V embed_dim=E
```

**Parameters:**
- `vocab_size` (required): Size of the vocabulary
- `embed_dim` (required): Embedding dimension

**Examples:**
```
embedding vocab_size=10000 embed_dim=512    # Standard transformer embedding
embedding vocab_size=50000 embed_dim=768    # Large vocabulary embedding
```

### Pooling Layer

Defines a pooling layer for downsampling.

```
pooling [type=TYPE] [size=S] [stride=ST]
```

**Parameters:**
- `type` (optional): Pooling type ('max', 'avg', 'global_avg'), default="max"
- `size` (optional): Pooling window size, default=2
- `stride` (optional): Pooling stride, default=2

**Examples:**
```
pooling                     # MaxPool 2x2, stride=2
pooling type=avg            # AvgPool 2x2, stride=2
pooling type=global_avg     # Global average pooling
pooling size=3 stride=1     # MaxPool 3x3, stride=1
```

### Batch Normalization

Applies batch normalization.

```
batchnorm
```

No parameters required.

## Complete Examples

### Image Classification CNN

```
input size=224x224x3
conv filters=32 kernel=3 stride=1 activation=relu
conv filters=32 kernel=3 stride=1 activation=relu
conv filters=64 kernel=3 stride=2 activation=relu
conv filters=64 kernel=3 stride=1 activation=relu
conv filters=128 kernel=3 stride=2 activation=relu
conv filters=128 kernel=3 stride=1 activation=relu
flatten
dense units=512 activation=relu
dropout rate=0.5
output units=1000 activation=softmax
```

### Simple MLP

```
input size=784
dense units=512 activation=relu
dropout rate=0.5
dense units=256 activation=relu
dropout rate=0.3
dense units=128 activation=relu
output units=10 activation=softmax
```

### Binary Classification

```
input size=100
dense units=64 activation=relu
dense units=32 activation=relu
dense units=16 activation=relu
output units=1 activation=sigmoid
```

### ResNet-style Architecture

Modern CNN with batch normalization and skip connections (conceptual).

```
input size=224x224x3
conv filters=64 kernel=7 stride=2
batchnorm
pooling type=max size=3 stride=2
conv filters=64 kernel=3
batchnorm
conv filters=64 kernel=3
batchnorm
conv filters=128 kernel=3 stride=2
batchnorm
conv filters=128 kernel=3
batchnorm
conv filters=256 kernel=3 stride=2
batchnorm
conv filters=256 kernel=3
batchnorm
conv filters=512 kernel=3 stride=2
batchnorm
conv filters=512 kernel=3
batchnorm
pooling type=global_avg
dense units=512
dropout rate=0.5
output units=1000
```

### Transformer Architecture

Simplified transformer for sequence classification.

```
input size=512
embedding vocab_size=10000 embed_dim=512
layernorm
attention heads=8 key_dim=64
layernorm
dense units=2048 activation=relu
dense units=512
attention heads=8 key_dim=64
layernorm
dense units=2048 activation=relu
dense units=512
pooling type=global_avg
dense units=256
dropout rate=0.1
output units=5
```

### Image Classification with Modern Techniques

CNN using batch normalization and different pooling strategies.

```
input size=224x224x3
conv filters=32 kernel=3
batchnorm
conv filters=32 kernel=3
batchnorm
pooling type=max size=2
conv filters=64 kernel=3
batchnorm
conv filters=64 kernel=3
batchnorm
pooling type=avg size=2
conv filters=128 kernel=3
batchnorm
conv filters=128 kernel=3
batchnorm
pooling type=global_avg
dense units=256
dropout rate=0.5
output units=10
```

## Using DSL in Python

### Basic Usage

```python
from neurink import Diagram

# Define network structure
dsl_text = """
input size=28x28
conv filters=32 kernel=3
flatten
dense units=128
output units=10
"""

# Create diagram from DSL
diagram = Diagram.from_string(dsl_text)

# Render to SVG
diagram.render("my_network.svg", theme="ieee")
```

### Loading from File

```python
# Save DSL to file
with open("network.dsl", "w") as f:
    f.write("""
    input size=64x64x3
    conv filters=64 kernel=7 stride=2
    conv filters=128 kernel=3
    conv filters=256 kernel=3 stride=2
    flatten
    dense units=1024
    dropout rate=0.5
    output units=1000
    """)

# Load and create diagram
with open("network.dsl", "r") as f:
    dsl_content = f.read()

diagram = Diagram.from_string(dsl_content)
diagram.render("loaded_network.svg")
```

## Activation Functions

Supported activation functions:
- `relu` (default for conv/dense)
- `tanh`
- `sigmoid`
- `softmax` (default for output)
- `linear`
- `leaky_relu`
- `elu`
- `swish`

## Best Practices

### 1. Consistent Formatting

Use consistent indentation and spacing for readability:

```
# Good
input size=224x224x3
conv filters=64 kernel=7 stride=2 activation=relu
conv filters=128 kernel=3 stride=1 activation=relu
flatten
dense units=512 activation=relu
output units=1000 activation=softmax

# Also acceptable
input      size=224x224x3
conv       filters=64  kernel=7 stride=2 activation=relu
conv       filters=128 kernel=3 stride=1 activation=relu
flatten
dense      units=512   activation=relu
output     units=1000  activation=softmax
```

### 2. Logical Layer Ordering

Follow the natural flow of data through the network:

```
input size=...      # Always start with input
# Feature extraction layers
conv ...
conv ...
# Dimensionality reduction
flatten
# Classification layers  
dense ...
dropout ...
output ...          # Always end with output
```

### 3. Meaningful Architectures

Design networks that make architectural sense:

```
# Good progression - increasing then decreasing feature maps
input size=224x224x3
conv filters=64 kernel=3    # Start with moderate features
conv filters=128 kernel=3   # Increase complexity
conv filters=256 kernel=3   # Peak complexity
flatten
dense units=512             # Reduce for classification
dense units=128             # Further reduce
output units=10             # Final classes
```

## Future Features (Roadmap)

The DSL will be enhanced with:

1. **Hierarchical Blocks**: Support for grouped layers
   ```
   encoder {
     conv filters=32 kernel=3
     conv filters=64 kernel=3
   }
   decoder {
     dense units=128
     output units=10
   }
   ```

2. **Comments**: Support for inline comments
   ```
   input size=28x28           # MNIST input
   conv filters=32 kernel=3   # First conv layer
   ```

3. **Variables**: Support for parameterized definitions
   ```
   @filters=32
   @kernel=3
   conv filters=@filters kernel=@kernel
   ```

4. **Skip Connections**: Support for ResNet-style connections
   ```
   input size=224x224x3
   conv filters=64 kernel=3 -> skip1
   conv filters=64 kernel=3
   add skip1
   ```

5. **Advanced Layers**: Support for attention, normalization, etc.
   ```
   attention heads=8 dim=512
   batchnorm
   layernorm
   ```

## Error Handling

Common DSL errors and solutions:

### Invalid Parameters
```
# Error: Unknown parameter
conv filters=32 invalid_param=value

# Solution: Check parameter names
conv filters=32 kernel=3
```

### Missing Required Parameters
```  
# Error: Missing required 'filters'
conv kernel=3

# Solution: Include all required parameters
conv filters=32 kernel=3
```

### Invalid Values
```
# Error: Invalid dropout rate
dropout rate=1.5

# Solution: Use valid range 0.0-1.0
dropout rate=0.5
```

For more examples and tutorials, see [TUTORIAL.md](TUTORIAL.md).