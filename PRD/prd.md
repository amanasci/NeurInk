I want you to build a Python library called NeurInk that generates publication-quality neural network diagrams. The library must be research-grade, user-friendly, highly customizable, and include a Python API, DSL parser, SVG rendering, tests, documentation, and project management features like a TODO list and task marking.

1️⃣ Project Structure (Mandatory)

Create the following Python project structure:

neurink/
  __init__.py
  diagram.py       # Core Diagram class
  layer.py         # Layer definitions
  parser.py        # DSL parser
  renderer.py      # SVG renderer
  themes.py        # Predefined style themes
  templates.py     # Prebuilt architectures (ResNet, UNet, Transformer)
  utils.py         # Helper functions
tests/
  test_diagram.py
  test_parser.py
  test_renderer.py
docs/
  README.md
  DSL.md
  TUTORIAL.md
setup.py / pyproject.toml
TODO.md           # Track remaining tasks and mark done as completed

2️⃣ Core Python API Requirements

Diagram class with methods:

input(shape)

conv(filters, kernel, stride=1, activation="relu")

dense(units, activation="relu")

flatten(), dropout(rate), output(units, activation)

render(filename, theme="ieee") → outputs SVG (optional PNG/PDF)

Must allow chaining:

diagram = Diagram().input((64,64)).conv(32,3).dense(128).output(10)
diagram.render("model.svg")


Include themes: ieee, apj, minimal, dark.

3️⃣ DSL / Markup Language

Lightweight DSL to define networks:

input size=64x64
conv filters=32 kernel=3
conv filters=64 kernel=3
flatten
dense units=128 activation=relu
output units=10 activation=softmax


Support hierarchical groups:

encoder {
  conv filters=32 kernel=3
  conv filters=64 kernel=3
}
decoder {
  dense units=128
  output units=10
}


Parser:

Use lark or pyparsing.

Convert DSL → internal Diagram object.

Provide from_string() API: Diagram.from_string(dsl_text).

4️⃣ Rendering

Render as clean, scalable SVG.

Optional exports: PNG, PDF.

Features:

Rounded boxes, clean spacing, arrow connections (curved/straight/dashed/weighted).

Support LaTeX-style annotations (MathJax or matplotlib text rendering).

Auto-layout: horizontal, vertical, encoder-decoder, custom spacing.

Templates for common networks: ResNet, UNet, Transformer, MLP.

5️⃣ Tests

Create unit tests for:

Diagram class methods (input, conv, dense, output, render).

DSL parser (from_string, hierarchical grouping, error handling).

Renderer (SVG structure, layers, connections, themes applied correctly).

Use pytest.

Tests should pass before marking any task done in TODO.md.

6️⃣ Documentation

README.md: installation, usage examples, API reference.

DSL.md: full DSL syntax, examples, tips.

TUTORIAL.md: walkthrough using Python API + DSL + rendering.

Inline docstrings in all Python classes and methods (Google-style or NumPy-style).

7️⃣ Project Management

Create TODO.md with tasks broken down into small actionable items:

Example:

- [ ] Create Diagram class
- [ ] Implement input() method
- [ ] Implement conv() method
- [ ] Implement dense() method
- [ ] Implement render() with SVG output
- [ ] Create DSL parser using lark/pyparsing
- [ ] Write tests for Diagram methods
- [ ] Write tests for DSL parser
- [ ] Write renderer tests
- [ ] Add themes
- [ ] Add template networks (ResNet, UNet)
- [ ] Write README.md
- [ ] Write DSL.md
- [ ] Write TUTORIAL.md


Mark [x] when task is complete.

After completing a task, update tests/documentation if necessary.

8️⃣ Coding Standards

Follow PEP8 and Python best practices.

Modular and maintainable code.

Clear naming conventions: Diagram, Layer, Renderer, etc.

9️⃣ Optional GUI Layer

Wrap in Streamlit or Gradio:

Left pane: DSL editor.

Right pane: real-time SVG preview.

Button to download SVG/PNG/PDF.

This is stretch goal, optional, mark as TODO.

10️⃣ Workflow Instructions for Agent

Break work into small tasks as per TODO.md.

After each task, mark it done and commit/test.

Update documentation incrementally.

Ensure each SVG output is visually clean and scalable.

Provide example scripts for both Python API and DSL usage.

💡 Goal: Create a fully functional Python library NeurInk for researchers to easily create publication-quality neural network diagrams, with DSL, themes, templates, tests, documentation, TODO tracking, and optional GUI. Every task should be testable, documented, and incremental, so the library is easy to maintain and extend.