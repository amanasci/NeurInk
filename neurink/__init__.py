"""
NeurInk: Publication-quality neural network diagram generation library.

A research-grade library for creating beautiful, customizable neural network
diagrams with Python API and DSL support.
"""

__version__ = "2.1.0"
__author__ = "NeurInk Contributors"

from .diagram import Diagram
from .layer import (
    Layer, InputLayer, ConvLayer, DenseLayer, FlattenLayer, DropoutLayer, OutputLayer,
    MaxPoolLayer, UpSampleLayer, ConvTransposeLayer, BatchNormLayer, LayerNormLayer,
    MultiHeadAttentionLayer, EmbeddingLayer, PositionalEncodingLayer, ReshapeLayer,
    GlobalAvgPoolLayer, ConcatenateLayer, AddLayer
)
from .renderer import SVGRenderer, GraphvizRenderer
from .themes import Theme, IEEETheme, APJTheme, MinimalTheme, DarkTheme
from .parser import DSLParser, DSLParseError
from .templates import ResNetTemplate, UNetTemplate, TransformerTemplate, MLPTemplate
from .blocks import BlockTemplate, get_block_registry

__all__ = [
    "Diagram",
    "Layer",
    "InputLayer",
    "ConvLayer", 
    "DenseLayer",
    "FlattenLayer",
    "DropoutLayer",
    "OutputLayer",
    "MaxPoolLayer",
    "UpSampleLayer",
    "ConvTransposeLayer",
    "BatchNormLayer",
    "LayerNormLayer",
    "MultiHeadAttentionLayer",
    "EmbeddingLayer",
    "PositionalEncodingLayer",
    "ReshapeLayer",
    "GlobalAvgPoolLayer",
    "ConcatenateLayer",
    "AddLayer",
    "SVGRenderer",
    "GraphvizRenderer",
    "Theme",
    "IEEETheme",
    "APJTheme",
    "MinimalTheme",
    "DarkTheme",
    "DSLParser",
    "DSLParseError",
    "ResNetTemplate",
    "UNetTemplate", 
    "TransformerTemplate",
    "MLPTemplate",
    "BlockTemplate",
    "get_block_registry",
]