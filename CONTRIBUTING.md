# Contributing to Peptide Agent

Thank you for your interest in contributing to the Peptide Agent project! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.10 or higher
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

## Development Workflow

### Code Quality

Before submitting a pull request, ensure your code passes all quality checks:

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=peptide_agent --cov-report=html

# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Type check (optional but recommended)
mypy src/
```

### Pre-commit Hooks

Pre-commit hooks will automatically run before each commit to ensure code quality. If a hook fails:

1. Review the error messages
2. Fix the issues
3. Stage the fixed files
4. Commit again

To run pre-commit hooks manually:
```bash
pre-commit run --all-files
```

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-new-model-support`
- `bugfix/fix-validation-error`
- `docs/update-readme`
- `refactor/improve-error-handling`

### Commit Messages

Write clear, descriptive commit messages:
- Use the imperative mood ("Add feature" not "Added feature")
- Keep the first line under 72 characters
- Add detailed explanation in the body if needed

Example:
```
Add validation for peptide synthesis conditions

- Implement Pydantic models for structured output
- Add parsing for interval notation
- Include comprehensive tests for edge cases
```

### Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Ensure all tests pass:**
   ```bash
   pytest
   ```

4. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a pull request:**
   - Provide a clear description of the changes
   - Reference any related issues
   - Request review from maintainers

### Pull Request Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main

## Testing Guidelines

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Name test functions with `test_` prefix
- Use descriptive test names that explain what is being tested

Example:
```python
def test_interval_bound_parsing_with_negative_numbers():
    """Test that IntervalBound correctly parses negative numbers."""
    interval = IntervalBound.from_string("(-3.0,-1.0)")
    assert interval.lower == -3.0
    assert interval.upper == -1.0
```

### Test Categories

- **Unit tests**: Test individual functions/classes in isolation
- **Integration tests**: Test multiple components working together
- **End-to-end tests**: Test complete user workflows

Mark integration tests appropriately:
```python
@pytest.mark.integration
def test_full_prediction_pipeline():
    # Test code here
    pass
```

## Code Style

### Python Style Guide

- Follow PEP 8
- Maximum line length: 100 characters
- Use type hints where appropriate
- Write docstrings for public functions and classes

### Docstring Format

Use Google-style docstrings:

```python
def predict_single(peptide_code: str, target_structural_assembly: str, settings: Settings) -> str:
    """Generate synthesis prediction for a single peptide.

    Args:
        peptide_code: Peptide sequence code (e.g., 'FF')
        target_structural_assembly: Target morphology (e.g., 'nanofibers')
        settings: Configuration settings

    Returns:
        Prediction report as string

    Raises:
        PeptideAgentError: If prediction fails
    """
    pass
```

## Reporting Issues

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (Python version, OS, etc.)
- Error messages and stack traces

### Feature Requests

Include:
- Clear description of the feature
- Use case / motivation
- Proposed implementation (if any)
- Examples of how it would be used

## Documentation

### Updating Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Update configuration documentation

### Building Documentation

If the project grows to include Sphinx documentation:
```bash
cd docs
make html
```

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

### Getting Help

- Check existing issues and documentation first
- Ask questions in GitHub issues
- Tag issues appropriately (`question`, `help-wanted`, etc.)

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
