# Contributing to FluxInfer

First off, thank you for considering contributing to FluxInfer. It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

FluxInfer is an ambitious project with the goal of becoming the **standard optimization engine for multimodal AI**. To achieve this, we maintain rigorous standards for code quality, performance, and documentation.

## 🧪 The FluxInfer Philosophy

1.  **Performance First**: Every abstraction must be zero-cost. If a feature adds latency, it must be behind a feature flag.
2.  **Rust Core, Python Interface**: We do the heavy lifting in Rust. Python is strictly for the API surface and orchestration.
3.  **Mathematical Rigor**: Optimizations (like Speculative Decoding) must be backed by correct statistical implementation.
4.  **Multimodal by Design**: Always consider how changes affect non-text modalities (Image, Audio).

## 🛠️ Development Setup

FluxInfer uses a hybrid Rust/Python workflow. You will need:
- **Rust 1.75+** (`cargo`)
- **Python 3.9+**
- **Maturin** (for building the bridge)

```bash
# 1. Clone the repository
git clone https://github.com/FluxInfer/FluxInfer.git
cd FluxInfer

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install development dependencies
pip install -r requirements.txt
pip install maturin pytest black isort mypy

# 4. Build the Rust core in development mode
maturin develop
```

## 🏗️ Project Structure

- `flux_infer_core/`: The Rust implementation (Kernels, Memory Management).
- `flux_infer/`: The Python SDK (Routing, API).
- `benchmarks/`: Performance verification scripts.
- `examples/`: Reference implementations.

## 📏 Code Style

We enforce strict linting to ensure codebase consistency.

### Python
We use **Black** and **Isort**.
```bash
# Run formatter
black .
isort .

# Run type checking
mypy flux_infer
```

### Rust
We use **rustfmt** and **clippy**.
```bash
# Run formatter
cargo fmt

# Run linter
cargo clippy -- -D warnings
```

## 🔄 Pull Request Process

1.  **Fork the repo** and create your branch from `main`.
2.  **Add tests** for any new features or bug fixes. We aim for 90%+ coverage.
3.  **Run benchmarks** (`python benchmarks/benchmark_suite.py`) to ensure no performance regressions.
4.  **Update documentation** if you change APIs.
5.  **Sign your commits** (GPG signature recommended).

## 🐛 Reporting Bugs

Please use the GitHub Issue Tracker. Include:
- Your OS and Hardware Specs (e.g., H100, A100, M2 Mac).
- Python and Rust versions.
- A minimal reproduction script.

## 📜 License

By contributing, you agree that your contributions will be licensed under its Apache 2.0 License.
