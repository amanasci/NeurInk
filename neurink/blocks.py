"""
Block template definitions for reusable neural network components.

This module provides a system for defining and using reusable block templates
that can be instantiated with parameters for complex architectures.
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class BlockTemplate(ABC):
    """Base class for reusable block templates."""
    
    def __init__(self, name: str):
        """Initialize a block template.
        
        Args:
            name: Name of the block template
        """
        self.name = name
    
    @abstractmethod
    def expand(self, **params) -> str:
        """Expand the block template with given parameters.
        
        Args:
            **params: Parameters to substitute in the template
            
        Returns:
            Expanded DSL text for the block
        """
        pass
    
    @abstractmethod 
    def get_required_params(self) -> List[str]:
        """Get list of required parameters for this block template.
        
        Returns:
            List of required parameter names
        """
        pass
    
    @abstractmethod
    def get_optional_params(self) -> Dict[str, Any]:
        """Get dict of optional parameters with their default values.
        
        Returns:
            Dict mapping parameter names to default values
        """
        pass


class ResidualBlockTemplate(BlockTemplate):
    """Template for a residual block (ResNet-style)."""
    
    def __init__(self):
        super().__init__("residual")
    
    def expand(self, **params) -> str:
        """Expand a residual block template."""
        filters = params.get('filters', 64)
        kernel_size = params.get('kernel_size', 3)
        block_name = params.get('block_name', params.get('name', 'residual'))
        
        return f"""
{block_name} {{
    conv filters={filters} kernel={kernel_size} name=conv1
    batch_norm name=bn1
    conv filters={filters} kernel={kernel_size} activation=linear name=conv2
    batch_norm name=bn2
}}
connect from={block_name}_conv1 to={block_name}_bn1
connect from={block_name}_bn1 to={block_name}_conv2
connect from={block_name}_conv2 to={block_name}_bn2
# Skip connection
connect from={block_name}_conv1 to={block_name}_bn2 type=skip style=dashed
"""
    
    def get_required_params(self) -> List[str]:
        return []
    
    def get_optional_params(self) -> Dict[str, Any]:
        return {
            'filters': 64,
            'kernel_size': 3,
            'name': 'residual'
        }


class AttentionBlockTemplate(BlockTemplate):
    """Template for a multi-head attention block (Transformer-style)."""
    
    def __init__(self):
        super().__init__("attention")
    
    def expand(self, **params) -> str:
        """Expand an attention block template."""
        num_heads = params.get('num_heads', 8)
        key_dim = params.get('key_dim', 64)
        block_name = params.get('block_name', params.get('name', 'attention'))
        
        return f"""
{block_name} {{
    multi_head_attention num_heads={num_heads} key_dim={key_dim} name=mha
    layer_norm name=ln1
    dense units={key_dim * num_heads} activation=relu name=ffn1
    dense units={key_dim * num_heads} activation=linear name=ffn2
    layer_norm name=ln2
}}
connect from={block_name}_mha to={block_name}_ln1
connect from={block_name}_ln1 to={block_name}_ffn1
connect from={block_name}_ffn1 to={block_name}_ffn2
connect from={block_name}_ffn2 to={block_name}_ln2
# Skip connections
connect from={block_name}_mha to={block_name}_ln2 type=skip style=dashed
"""
    
    def get_required_params(self) -> List[str]:
        return []
    
    def get_optional_params(self) -> Dict[str, Any]:
        return {
            'num_heads': 8,
            'key_dim': 64,
            'name': 'attention'
        }


class EncoderBlockTemplate(BlockTemplate):
    """Template for an encoder block with customizable architecture."""
    
    def __init__(self):
        super().__init__("encoder")
    
    def expand(self, **params) -> str:
        """Expand an encoder block template."""
        filters = params.get('filters', [32, 64, 128])
        if not isinstance(filters, list):
            filters = [filters]
        
        block_name = params.get('block_name', params.get('name', 'encoder'))
        kernel_size = params.get('kernel_size', 3)
        use_pooling = params.get('use_pooling', True)
        
        dsl_lines = [f"{block_name} {{"]
        
        for i, f in enumerate(filters):
            dsl_lines.append(f"    conv filters={f} kernel={kernel_size} name=conv{i+1}")
            if i < len(filters) - 1 or use_pooling:
                dsl_lines.append(f"    maxpool name=pool{i+1}")
        
        dsl_lines.append("}")
        
        return "\n".join(dsl_lines)
    
    def get_required_params(self) -> List[str]:
        return []
    
    def get_optional_params(self) -> Dict[str, Any]:
        return {
            'filters': [32, 64, 128],
            'kernel_size': 3,
            'use_pooling': True,
            'name': 'encoder'
        }


class BlockTemplateRegistry:
    """Registry for managing block templates."""
    
    def __init__(self):
        """Initialize the block template registry."""
        self._templates: Dict[str, BlockTemplate] = {}
        
        # Register built-in templates
        self.register(ResidualBlockTemplate())
        self.register(AttentionBlockTemplate())
        self.register(EncoderBlockTemplate())
    
    def register(self, template: BlockTemplate) -> None:
        """Register a block template.
        
        Args:
            template: Block template to register
        """
        self._templates[template.name] = template
    
    def get(self, name: str) -> Optional[BlockTemplate]:
        """Get a block template by name.
        
        Args:
            name: Name of the template
            
        Returns:
            Block template or None if not found
        """
        return self._templates.get(name)
    
    def list_templates(self) -> List[str]:
        """Get list of available template names.
        
        Returns:
            List of template names
        """
        return list(self._templates.keys())
    
    def expand_template(self, name: str, **params) -> str:
        """Expand a template with given parameters.
        
        Args:
            name: Template name
            **params: Template parameters
            
        Returns:
            Expanded DSL text
            
        Raises:
            ValueError: If template not found
        """
        template = self.get(name)
        if template is None:
            available = ", ".join(self.list_templates())
            raise ValueError(f"Block template '{name}' not found. Available templates: {available}")
        
        return template.expand(**params)


# Global registry instance
_block_registry = BlockTemplateRegistry()

def get_block_registry() -> BlockTemplateRegistry:
    """Get the global block template registry."""
    return _block_registry