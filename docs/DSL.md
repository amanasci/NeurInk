# NeurInk DSL Reference

The NeurInk Domain-Specific Language (DSL) provides a simple, readable way to define neural network architectures. This document covers the complete DSL syntax, examples, and best practices.

**New in v2.0:** Support for named layers and explicit connections for complex architectures with skip connections and branching.

## Basic Syntax

The DSL uses a simple line-based format where each line defines a layer:

```
layer_type parameter1=value1 parameter2=value2 [name=layer_name]
```

### New in v2.0: Named Layers

All layers can now be named for explicit connections:

```
input size=224x224x3 name=input1
conv filters=64 kernel=3 name=conv_block1
dense units=128 name=classifier
```

If no name is provided, one will be automatically generated (e.g., `conv_1`, `dense_2`).

### New in v2.0: Connections

Create explicit connections between layers to build complex architectures:

```
connect from=layer1 to=layer2
```

This enables skip connections, residual blocks, and branching architectures.

### Comments and Whitespace

- Empty lines are ignored
- Leading and trailing whitespace is ignored
- Comments are supported with `#` (full line or inline)

**Examples:**
```
# This is a full line comment
input size=28x28 name=input  # This is an inline comment
conv filters=32 kernel=3     # Another inline comment
```

### New in v2.1: Hierarchical Blocks

Group related layers into hierarchical blocks for better organization and automatic naming.

```
block_name {
    layer_type parameter1=value1
    layer_type parameter2=value2
}
```

**Features:**
- Layers inside blocks get automatic prefixed names (`block_layer_type_N`)
- Nested blocks are supported
- Improves readability for complex architectures
- Maintains all layer functionality within blocks

**Examples:**
```
# Simple encoder-decoder architecture
encoder {
    conv filters=32 kernel=3
    conv filters=64 kernel=3
    maxpool pool_size=2
}

decoder {
    dense units=256
    dense units=128
    dropout rate=0.5
}

# Nested blocks for complex architectures
transformer {
    encoder {
        multi_head_attention heads=8 dim=512
        layer_norm
        dense units=2048
    }
    
    decoder {
        multi_head_attention heads=8 dim=512
        layer_norm
        dense units=2048
    }
}
```

### LaTeX Support in Labels

Layer display names support basic LaTeX mathematical symbols and expressions.

**Supported Features:**
- Greek letters: `\alpha`, `\beta`, `\gamma`, etc. → α, β, γ
- Mathematical symbols: `\partial`, `\nabla`, `\sum`, etc. → ∂, ∇, ∑
- Subscripts and superscripts: `x_1`, `x^2` → x₁, x²

**Examples:**
```
conv filters=32 kernel=3 display_name="Conv2D_\alpha"
dense units=128 display_name="FC_\theta" 
output units=10 display_name="Softmax_\pi"
```

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

### MaxPool Layer

Max pooling layer for downsampling spatial dimensions.

```
maxpool [pool_size=N] [stride=N]
```

**Parameters:**
- `pool_size` (optional): Size of pooling window, default=2
- `stride` (optional): Stride of pooling operation, default=same as pool_size

**Examples:**
```
maxpool                    # 2x2 pooling
maxpool pool_size=3        # 3x3 pooling
maxpool pool_size=2 stride=1  # 2x2 pooling with stride 1
```

### UpSample Layer

Upsampling layer for increasing spatial resolution.

```
upsample [size=N] [method=METHOD]
```

**Parameters:**
- `size` (optional): Upsampling factor, default=2
- `method` (optional): Upsampling method ('nearest', 'bilinear'), default="nearest"

**Examples:**
```
upsample                   # 2x upsampling with nearest neighbor
upsample size=4            # 4x upsampling
upsample size=2 method=bilinear  # Bilinear upsampling
```

### ConvTranspose Layer

Transposed convolution (deconvolution) layer.

```
conv_transpose filters=N kernel=K [stride=S] [activation=ACT]
```

**Parameters:**
- `filters` (required): Number of output filters
- `kernel` (required): Size of convolution kernel
- `stride` (optional): Stride of convolution, default=1
- `activation` (optional): Activation function, default="relu"

**Examples:**
```
conv_transpose filters=64 kernel=2 stride=2    # Upsampling convolution
conv_transpose filters=32 kernel=3             # Regular transposed conv
```

### Batch Normalization Layer

Batch normalization for training stability.

```
batch_norm
```

No parameters required.

### Layer Normalization Layer

Layer normalization for transformer architectures.

```
layer_norm
```

No parameters required.

### Multi-Head Attention Layer

Multi-head attention mechanism for transformers.

```
multi_head_attention num_heads=N key_dim=D
```

**Parameters:**
- `num_heads` (required): Number of attention heads
- `key_dim` (required): Dimension of attention keys/queries

**Examples:**
```
multi_head_attention num_heads=8 key_dim=64    # Standard transformer attention
multi_head_attention num_heads=12 key_dim=128  # Larger attention
```

### Embedding Layer

Token embedding layer for transformers.

```
embedding vocab_size=V embed_dim=D
```

**Parameters:**
- `vocab_size` (required): Size of vocabulary
- `embed_dim` (required): Embedding dimension

**Examples:**
```
embedding vocab_size=10000 embed_dim=512   # Word embeddings
embedding vocab_size=1000 embed_dim=256    # Smaller vocabulary
```

### Positional Encoding Layer

Positional encoding for transformer inputs.

```
positional_encoding max_len=L embed_dim=D
```

**Parameters:**
- `max_len` (required): Maximum sequence length
- `embed_dim` (required): Embedding dimension

**Examples:**
```
positional_encoding max_len=512 embed_dim=512  # Standard transformer
positional_encoding max_len=1024 embed_dim=768 # Longer sequences
```

### Reshape Layer

Reshape tensor dimensions.

```
reshape shape=SHAPE
```

**Parameters:**
- `shape` (required): Target shape (e.g., "64x64" or "128x8x8")

**Examples:**
```
reshape shape=64x64        # Reshape to 2D
reshape shape=128x8x8      # Reshape to 3D
```

### Global Average Pooling Layer

Global average pooling across spatial dimensions.

```
global_avg_pool
```

No parameters required.

### Concatenate Layer

Concatenate multiple inputs along specified axis.

```
concatenate [axis=N]
```

**Parameters:**
- `axis` (optional): Concatenation axis, default=-1

**Examples:**
```
concatenate            # Concatenate on last axis
concatenate axis=1     # Concatenate on axis 1
```

### Add Layer

Element-wise addition for residual connections.

```
add
```

No parameters required.

### Connections (Enhanced in v2.1)

Create explicit connections between layers to build complex architectures with enhanced visual styling and semantic types.

```
connect from=SOURCE_LAYER to=TARGET_LAYER [type=TYPE] [style=STYLE] [weight=WEIGHT] [label=LABEL]
```

**Parameters:**
- `from` (required): Name of the source layer
- `to` (required): Name of the target layer  
- `type` (optional): Connection type - `default`, `skip`, `residual`, `attention`, `feedback`
- `style` (optional): Visual style - `solid`, `dashed`, `dotted`, `bold`
- `weight` (optional): Connection weight (affects line thickness, ≥0.0)
- `label` (optional): Text label for the connection

**Connection Types:**
- `default`: Standard connection (black)
- `skip`: Skip connections (blue) 
- `residual`: Residual connections (orange)
- `attention`: Attention mechanisms (purple)
- `feedback`: Feedback connections (red)

**Examples:**
```
# Basic skip connection
connect from=conv1 to=conv3 type=skip style=dashed

# Weighted residual connection
connect from=input to=output_block type=residual weight=0.5 style=bold

# Attention connection with label
connect from=encoder to=decoder type=attention style=dotted label="Attn"

# Multi-input fusion with different weights
connect from=branch1 to=fusion_layer weight=0.6
connect from=branch2 to=fusion_layer weight=0.4
```

**Note:** Connections create additional edges in the network graph. The normal sequential flow is preserved automatically.

## Complete Examples

## Advanced Examples (v2.0)

### ResNet-style Architecture with Skip Connections

```
input size=224x224x3 name=input
conv filters=64 kernel=7 stride=2 name=conv1

# First residual block
conv filters=64 kernel=1 activation=relu name=conv2_1x1
conv filters=64 kernel=3 activation=relu name=conv2_3x3  
conv filters=256 kernel=1 activation=linear name=conv2_out
connect from=conv1 to=conv2_out

# Second residual block
conv filters=64 kernel=1 activation=relu name=conv3_1x1
conv filters=64 kernel=3 activation=relu name=conv3_3x3
conv filters=256 kernel=1 activation=linear name=conv3_out
connect from=conv2_out to=conv3_out

flatten name=avgpool
dense units=512 name=fc
output units=1000 name=classifier
```

### U-Net Style Architecture with Skip Connections

```
input size=256x256x3 name=input

# Encoder
conv filters=64 kernel=3 name=conv1_1
conv filters=64 kernel=3 name=conv1_2
conv filters=128 kernel=3 stride=2 name=conv2_1
conv filters=128 kernel=3 name=conv2_2
conv filters=256 kernel=3 stride=2 name=conv3_1
conv filters=256 kernel=3 name=conv3_2

# Decoder with skip connections
conv filters=128 kernel=3 name=up_conv2
connect from=conv2_2 to=up_conv2
conv filters=64 kernel=3 name=up_conv1  
connect from=conv1_2 to=up_conv1
conv filters=3 kernel=1 name=output_conv
```

### Multi-Input Network

```
input size=224x224x3 name=image_input
input size=100 name=metadata_input

# Image processing branch
conv filters=32 kernel=3 name=img_conv1
conv filters=64 kernel=3 name=img_conv2
flatten name=img_flatten
dense units=256 name=img_features

# Metadata processing branch  
dense units=128 name=meta_dense1
dense units=256 name=meta_features

# Combine branches
dense units=512 name=combined
connect from=img_features to=combined
connect from=meta_features to=combined

output units=10 name=prediction
```

## Basic Examples

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