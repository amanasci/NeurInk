# NeurInk v2.0 Development TODO

This document tracks the development progress and future roadmap for NeurInk v2.0, which introduces graph-based architecture support for complex, non-linear neural networks.

## Completed in v2.0 ✅

### Core Architecture Upgrade
- [x] **Graph-Based Foundation**
  - [x] Replace list-based layer storage with NetworkX DiGraph
  - [x] Implement automatic layer naming system
  - [x] Add `connect(source, target)` method for explicit connections
  - [x] Maintain backward compatibility with existing APIs
  - [x] Add `get_layer_names()` method for introspection

### Enhanced DSL Parser
- [x] **Named Layers Support**
  - [x] Support `name=layer_name` parameter in all layer types
  - [x] Automatic name generation when not specified
  - [x] Connection syntax: `connect from=source to=target`
  - [x] Backward compatibility with existing DSL files
  - [x] Comprehensive error handling and validation

### Advanced Rendering Engine
- [x] **Graphviz Integration**
  - [x] Replace manual SVG generation with Graphviz
  - [x] HTML-like labels with detailed layer information
  - [x] Support for complex graph layouts (skip connections, branching)
  - [x] Theme system integration with new renderer
  - [x] Backward compatibility for template rendering

### Testing & Quality Assurance
- [x] **Comprehensive Test Coverage**
  - [x] All 87 existing tests pass without modification
  - [x] Graph functionality tests
  - [x] Named layer tests  
  - [x] Connection method tests
  - [x] Enhanced DSL parser tests
  - [x] Rendering compatibility tests

### Documentation & Examples
- [x] **Updated Documentation**
  - [x] Enhanced README.md with v2.0 features
  - [x] Updated DSL.md with new syntax and examples
  - [x] Created comprehensive example_v2.py
  - [x] ResNet-style architecture examples
  - [x] Skip connection demonstrations

## Future Enhancements (v2.1+)

### Advanced DSL Features
- [ ] **Hierarchical Grouping**
  - [ ] Support for nested blocks: `encoder { ... } decoder { ... }`
  - [ ] Group-level connections and operations
  - [ ] Named group references
  - [ ] Block templates and reusability

- [ ] **Enhanced Syntax**
  - [ ] Comment support with `#` or `//`
  - [ ] Multi-line parameter definitions
  - [ ] Variable definitions and substitutions
  - [ ] Conditional layer inclusion

### Advanced Graph Operations
- [ ] **Graph Algorithms**
  - [ ] Automatic layout optimization for complex graphs
  - [ ] Cycle detection and validation
  - [ ] Graph simplification and merging
  - [ ] Subgraph extraction and composition

- [ ] **Advanced Connection Types**
  - [ ] Weighted connections with visual representation
  - [ ] Connection labels and annotations  
  - [ ] Conditional connections
  - [ ] Multi-output and multi-input explicit handling

### Rendering Improvements
- [ ] **Layout Enhancements**
  - [ ] Hierarchical layouts for deep networks
  - [ ] Circular layouts for recurrent architectures
  - [ ] Custom positioning hints and constraints
  - [ ] Interactive SVG with hover information

- [ ] **Export Formats**
  - [ ] PNG export with high DPI support
  - [ ] PDF export for publications
  - [ ] LaTeX/TikZ export for academic papers
  - [ ] Interactive HTML export

### Advanced Architecture Support
- [ ] **Specialized Layer Types**
  - [ ] Attention layers (self-attention, cross-attention)
  - [ ] Normalization layers (BatchNorm, LayerNorm, GroupNorm)
  - [ ] Pooling layers (MaxPool, AvgPool, AdaptivePool)
  - [ ] Recurrent layers (LSTM, GRU, RNN)
  - [ ] Transformer blocks

- [ ] **Architecture Templates**
  - [ ] Vision Transformer (ViT) template
  - [ ] BERT/GPT-style transformer templates
  - [ ] GAN architectures (Generator/Discriminator)
  - [ ] Autoencoder templates
  - [ ] Custom template definition system

### Developer Experience
- [ ] **Tooling & IDE Support**
  - [ ] VS Code extension for DSL syntax highlighting
  - [ ] Language server protocol support
  - [ ] Auto-completion for layer types and parameters
  - [ ] Real-time validation and error checking

- [ ] **CLI Tools**
  - [ ] Command-line diagram generation
  - [ ] Batch processing of DSL files
  - [ ] Format conversion utilities
  - [ ] Validation and linting tools

### Performance & Scalability  
- [ ] **Large Network Support**
  - [ ] Efficient handling of networks with 1000+ layers
  - [ ] Lazy evaluation for large graphs  
  - [ ] Streaming rendering for memory efficiency
  - [ ] Parallel processing support

- [ ] **Caching & Optimization**
  - [ ] Layout caching for repeated renders
  - [ ] Incremental updates for interactive editing
  - [ ] Memory usage optimization
  - [ ] Rendering performance profiling

## Optional/Experimental Features

### GUI Applications
- [ ] **Desktop Application**
  - [ ] Standalone GUI for network design
  - [ ] Drag-and-drop interface
  - [ ] Real-time preview
  - [ ] Export management

- [ ] **Web Interface**  
  - [ ] Browser-based editor
  - [ ] Collaborative editing
  - [ ] Cloud-based rendering
  - [ ] Gallery of public diagrams

### Integration & Ecosystem
- [ ] **Framework Integration**
  - [ ] PyTorch model introspection
  - [ ] TensorFlow/Keras model import
  - [ ] ONNX model visualization
  - [ ] Hugging Face model support

- [ ] **Research Tools**
  - [ ] Jupyter notebook magic commands
  - [ ] Streamlit/Gradio components
  - [ ] Paper template integration
  - [ ] Citation management

## Development Guidelines

### Contributing
- Maintain 100% backward compatibility unless major version bump
- All new features must include comprehensive tests
- Documentation must be updated for any API changes
- Follow existing code style and conventions

### Release Process
- Minor versions (v2.x) for new features with backward compatibility
- Patch versions (v2.x.y) for bug fixes only
- Major versions (v3.0) for breaking changes only
- Alpha/Beta releases for experimental features

### Testing Requirements
- Unit tests for all new functionality
- Integration tests for complex features
- Backward compatibility tests for all changes
- Performance regression tests for core features

## Notes
- This TODO represents the roadmap and may be adjusted based on user feedback
- Feature priorities may change based on community requests
- Contributors are welcome to pick up any unclaimed items
- See CONTRIBUTING.md for development setup and guidelines