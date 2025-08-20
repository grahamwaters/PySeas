# Contributing to PySeas 🤝

**Thank you for your interest in contributing to PySeas!**

## 🎯 How to Contribute

### Types of Contributions

- 🐛 **Bug Reports**: Report issues and bugs
- 💡 **Feature Requests**: Suggest new features
- 📝 **Documentation**: Improve docs and examples
- 🔧 **Code**: Fix bugs or implement features
- 🧪 **Testing**: Add or improve tests
- 🌐 **Localization**: Translate to other languages

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git
- Basic knowledge of Python and image processing

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/PySeas.git
cd PySeas

# Create development environment
python -m venv pyseas_dev
source pyseas_dev/bin/activate  # On Windows: pyseas_dev\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify setup
python -m pytest tests/ --version
```

## 🔧 Development Workflow

### 1. Create Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/amazing-feature
```

### 2. Make Changes

- Follow the [code standards](DEVELOPER_GUIDE.md#code-standards)
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and descriptive

### 3. Test Your Changes

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/

# Check code style
black src/
flake8 src/
mypy src/
```

### 4. Commit and Push

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add amazing feature: brief description"

# Push to your fork
git push origin feature/amazing-feature
```

### 5. Create Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select your feature branch
4. Fill out the PR template
5. Submit for review

## 📝 Code Standards

### Python Style Guide

- Follow **PEP 8** with project-specific rules
- Use **Black** for code formatting
- Use **flake8** for linting
- Use **mypy** for type checking

### Code Quality Checklist

- [ ] Code follows style guidelines
- [ ] Functions have docstrings
- [ ] Type hints are used
- [ ] Error handling is appropriate
- [ ] Logging is informative
- [ ] No hardcoded values

### Example Good Code

```python
def process_buoy_image(image_path: str, quality_threshold: float = 0.7) -> ProcessingResult:
    """
    Process a single buoy image.

    Args:
        image_path: Path to the image file
        quality_threshold: Minimum quality score (0.0-1.0)

    Returns:
        ProcessingResult containing results and metadata

    Raises:
        FileNotFoundError: If image file doesn't exist
        ProcessingError: If image processing fails
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        # Process image
        result = _process_image(image_path)

        # Check quality
        if result.quality_score < quality_threshold:
            logger.warning(f"Low quality image: {result.quality_score}")

        return result

    except Exception as e:
        logger.error(f"Failed to process {image_path}: {e}")
        raise ProcessingError(f"Image processing failed: {e}")
```

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_image_processing.py
│   └── test_data_collection.py
├── integration/             # Integration tests
│   └── test_full_pipeline.py
└── fixtures/                # Test data
    └── sample_images/
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch
from src.image_processor import ImageProcessor

class TestImageProcessor:
    """Test cases for ImageProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create ImageProcessor instance for testing."""
        return ImageProcessor(quality_threshold=0.7)

    def test_quality_assessment_valid_image(self, processor):
        """Test quality assessment with valid image."""
        # Arrange
        test_image = create_test_image((100, 100))

        # Act
        quality_score = processor.assess_quality(test_image)

        # Assert
        assert 0.0 <= quality_score <= 1.0
        assert quality_score > 0.5

    def test_quality_assessment_invalid_input(self, processor):
        """Test quality assessment with invalid input."""
        # Act & Assert
        with pytest.raises(ValueError, match="Image cannot be None"):
            processor.assess_quality(None)
```

### Test Coverage

- Aim for **90%+ code coverage**
- Test both success and failure cases
- Test edge cases and boundary conditions
- Use mocking for external dependencies

## 📚 Documentation

### Updating Documentation

- Update relevant documentation files
- Add examples for new features
- Update API reference if needed
- Include usage examples

### Documentation Standards

- Use clear, concise language
- Include code examples
- Use proper markdown formatting
- Keep documentation up to date

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues
2. Search documentation
3. Try to reproduce the issue
3. Check system requirements

### Bug Report Template

```markdown
**Bug Description**
Brief description of the issue

**Steps to Reproduce**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.10.0]
- PySeas: [e.g., 1.0.0]

**Additional Context**
Any other relevant information
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
Brief description of the feature

**Use Case**
Why this feature would be useful

**Proposed Implementation**
How you think it could be implemented

**Alternatives Considered**
Other approaches you've considered

**Additional Context**
Any other relevant information
```

## 🔍 Code Review Process

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass and cover new functionality
- [ ] Documentation is updated
- [ ] No security vulnerabilities introduced
- [ ] Performance impact considered
- [ ] Error handling is appropriate
- [ ] Logging is informative
- [ ] Dependencies are minimal and justified

### Review Guidelines

- Be constructive and respectful
- Focus on code quality and functionality
- Suggest improvements when possible
- Ask questions if something is unclear

## 🚀 Release Process

### Versioning

PySeas follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Version numbers are updated
- [ ] Release notes are prepared

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions
- **Email**: Direct contact for sensitive issues

### Community Guidelines

- Be respectful and inclusive
- Help others when possible
- Follow the project's code of conduct
- Report inappropriate behavior

## 🙏 Recognition

### Contributors

All contributors are recognized in:
- GitHub contributors list
- Project documentation
- Release notes
- Project website

### Types of Recognition

- **Code Contributors**: Listed in contributors
- **Documentation**: Acknowledged in docs
- **Bug Reports**: Thanked in issue resolution
- **Feature Ideas**: Credited in implementation

---

## 📋 Quick Reference

### Common Commands

```bash
# Setup development environment
python -m venv pyseas_dev
source pyseas_dev/bin/activate
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black src/
flake8 src/

# Type checking
mypy src/

# Install pre-commit hooks
pre-commit install
```

### Useful Links

- [Developer Guide](DEVELOPER_GUIDE.md)
- [API Reference](API_REFERENCE.md)
- [Architecture](ARCHITECTURE.md)
- [Code of Conduct](CODE_OF_CONDUCT.md)

---

**Thank you for contributing to PySeas! 🌊**

*Your contributions help make ocean exploration accessible to everyone.*
