# NeurInk Development TODO

This document tracks the development progress of the NeurInk library for generating publication-quality neural network diagrams.

## Phase 1: Project Structure & Core Foundation
- [x] Create project directory structure (neurink/, tests/, docs/)
- [x] Create pyproject.toml with dependencies
- [x] Create TODO.md with detailed task breakdown
- [x] Set up basic project files (.gitignore, etc.)
- [x] Create basic neurink package structure

## Phase 2: Core Library Components
- [x] Implement Layer base classes and layer definitions
- [x] Implement core Diagram class with method chaining
- [x] Implement input() method
- [x] Implement conv() method  
- [x] Implement dense() method
- [x] Implement flatten(), dropout(), output() methods
- [x] Create SVG renderer with basic functionality
- [x] Implement render() method with SVG output

## Phase 3: DSL Parser
- [x] Choose and implement DSL parser (basic implementation)
- [x] Support basic DSL syntax (input, conv, dense, etc.)
- [ ] Support hierarchical groups in DSL (future enhancement)
- [x] Implement Diagram.from_string() API
- [x] Add error handling for invalid DSL
- [ ] Enhance DSL parser with lark/pyparsing for complex syntax (future enhancement)

## Phase 4: Themes & Styling
- [x] Create theme system with base Theme class
- [x] Implement ieee theme
- [x] Implement apj theme
- [x] Implement minimal theme
- [x] Implement dark theme

## Phase 5: Templates & Advanced Features
- [x] Create template system for common architectures
- [x] Implement ResNet template
- [x] Implement UNet template
- [x] Implement Transformer template
- [x] Implement MLP template
- [x] Add utility functions

## Phase 6: Testing
- [x] Write tests for Diagram class methods
- [x] Write tests for DSL parser functionality
- [x] Write tests for SVG renderer
- [x] Write tests for themes
- [x] Write tests for templates
- [x] Ensure all tests pass (87/87 tests passing ✅)

## Phase 7: Documentation
- [x] Write comprehensive README.md
- [x] Write DSL.md documentation
- [x] Write TUTORIAL.md walkthrough
- [x] Add inline docstrings to all classes/methods
- [x] Create example scripts

## Phase 8: Optional Features
- [ ] Add PNG/PDF export capabilities
- [ ] Create optional GUI with Streamlit/Gradio
- [ ] Performance optimizations

## Notes
- Tests should pass before marking any task as [x] done
- Update documentation incrementally after each task
- Ensure SVG output is clean and scalable
- Follow PEP8 and Python best practices