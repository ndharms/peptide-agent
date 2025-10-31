# Contributing to Peptide Agent

Thank you for your interest in contributing to the Peptide Agent project! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- A Gemini API key for testing

### Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/peptide-agent.git
   cd peptide-agent
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

5. **Set up environment variables:**
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```

## Release Process

Releases are handled by maintainers:

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create a GitHub release
4. CI/CD automatically publishes to PyPI and Docker Hub

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the `question` label
- Reach out to the maintainers

Thank you for contributing to Peptide Agent! 🧬
