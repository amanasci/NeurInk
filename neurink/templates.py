"""
Template architectures for common neural networks.

Provides pre-built templates for popular architectures like
ResNet, UNet, Transformer, and MLP.
"""

from .diagram import Diagram


class NetworkTemplate:
    """Base class for network architecture templates."""
    
    @staticmethod
    def create() -> Diagram:
        """Create and return a diagram with the template architecture."""
        raise NotImplementedError


class ResNetTemplate(NetworkTemplate):
    """ResNet-style architecture template."""
    
    @staticmethod
    def create(input_shape: tuple = (224, 224, 3), num_classes: int = 1000) -> Diagram:
        """
        Create a simplified ResNet-style architecture.
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
            
        Returns:
            Diagram with ResNet-like architecture
        """
        return (Diagram()
                .input(input_shape)
                .conv(64, 7, stride=2)
                .batch_norm()
                .pooling("max", pool_size=3, stride=2)
                .conv(64, 3)
                .batch_norm()
                .conv(64, 3)
                .batch_norm()
                .conv(128, 3, stride=2)
                .batch_norm() 
                .conv(128, 3)
                .batch_norm()
                .conv(256, 3, stride=2)
                .batch_norm()
                .conv(256, 3)
                .batch_norm()
                .conv(512, 3, stride=2)
                .batch_norm()
                .conv(512, 3)
                .batch_norm()
                .pooling("global_avg")
                .dense(512)
                .dropout(0.5)
                .output(num_classes))


class UNetTemplate(NetworkTemplate):
    """UNet architecture template for segmentation."""
    
    @staticmethod
    def create(input_shape: tuple = (256, 256, 3), num_classes: int = 1) -> Diagram:
        """
        Create a simplified UNet architecture.
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes/channels
            
        Returns:
            Diagram with UNet-like architecture
        """
        # Note: This is a simplified representation
        # Real UNet has skip connections which aren't fully supported yet
        return (Diagram()
                .input(input_shape)
                .conv(64, 3)
                .conv(64, 3)
                .conv(128, 3, stride=2)
                .conv(128, 3)
                .conv(256, 3, stride=2)
                .conv(256, 3)
                .conv(512, 3, stride=2)
                .conv(512, 3)
                .conv(256, 3)
                .conv(256, 3)
                .conv(128, 3)
                .conv(128, 3)
                .conv(64, 3)
                .conv(64, 3)
                .output(num_classes, activation="sigmoid"))


class TransformerTemplate(NetworkTemplate):
    """Transformer architecture template."""
    
    @staticmethod
    def create(vocab_size: int = 10000, max_length: int = 512, 
               num_classes: int = 2) -> Diagram:
        """
        Create a simplified Transformer architecture.
        
        Args:
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
            num_classes: Number of output classes
            
        Returns:
            Diagram with Transformer-like architecture
        """
        # More realistic Transformer representation
        return (Diagram()
                .input((max_length,))
                .embedding(vocab_size, 512)
                .layer_norm()
                .attention(num_heads=8, key_dim=64)
                .layer_norm()
                .dense(2048, activation="relu")  # Feed forward
                .dense(512, activation="relu")
                .attention(num_heads=8, key_dim=64)
                .layer_norm()
                .dense(2048, activation="relu")  # Feed forward  
                .dense(512, activation="relu")
                .pooling("global_avg")
                .dense(256)
                .dropout(0.1)
                .output(num_classes))


class MLPTemplate(NetworkTemplate):
    """Multi-Layer Perceptron template."""
    
    @staticmethod  
    def create(input_size: int = 784, hidden_sizes: list = None, 
               num_classes: int = 10) -> Diagram:
        """
        Create an MLP architecture.
        
        Args:
            input_size: Size of input layer
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            
        Returns:
            Diagram with MLP architecture
        """
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]
            
        diagram = Diagram().input((input_size,))
        
        for size in hidden_sizes:
            diagram = diagram.dense(size).dropout(0.5)
            
        return diagram.output(num_classes)