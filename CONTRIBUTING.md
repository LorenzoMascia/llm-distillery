# Contributing to LLM Distillery - Data Anonymization

Thank you for your interest in contributing! This document provides guidelines for contributing to the LLM Distillery data anonymization project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/llm-distillery.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install development dependencies: `make install-dev`

## Development Workflow

1. Create a new branch for your feature: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `make test`
4. Run linters: `make lint`
5. Format code: `make format`
6. Commit your changes: `git commit -m "Description of changes"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Code Style

- We use [Black](https://black.readthedocs.io/) for code formatting (line length: 100)
- We use [isort](https://pycqa.github.io/isort/) for import sorting
- We use [flake8](https://flake8.pycqa.org/) for linting
- Run `make format` before committing

## Testing

- Write tests for new features
- Maintain or improve code coverage
- Run `make test` to execute the test suite
- Tests are located in the `tests/` directory

## Documentation

- Update [README.md](README.md) if adding new features
- Update [README_ANONYMIZATION.md](README_ANONYMIZATION.md) for anonymization-specific changes
- Add docstrings to all functions and classes (Google style)
- Update configuration examples if changing config structure
- Update [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) if modifying the project structure

## Contributing to Anonymization Features

### Adding New PII Types

To add support for new types of PII:

1. Add examples to `config/prompts_anonymization.yaml`:
   ```yaml
   examples:
     - input: "Text with new PII type..."
       output:
         anonymized_text: "Text with [NEW_TYPE_1]..."
         replaced_tokens:
           - replaced_value: "[NEW_TYPE_1]"
             original_value: "original value"
   ```

2. Generate new training data with the updated prompts
3. Test the model's ability to recognize the new PII type
4. Document the new PII type in README_ANONYMIZATION.md

### Improving Model Performance

To improve anonymization accuracy:

1. Add more diverse training examples
2. Experiment with different prompt formulations
3. Adjust LoRA hyperparameters
4. Test with different base models
5. Document your findings and improvements

### Adding Support for Other Languages

To add support for additional languages:

1. Add language-specific examples in `config/prompts_anonymization.yaml`
2. Ensure proper tokenization for the target language
3. Generate training data in the new language
4. Test and validate performance
5. Update documentation with language support details

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Update documentation as needed
- Keep PRs focused on a single feature or fix

## Reporting Issues

When reporting issues, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages (if any)

## Feature Requests

We welcome feature requests! Please:

- Check if the feature has already been requested
- Provide a clear use case
- Explain how it aligns with the project goals

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

## Questions?

Feel free to open an issue for questions or clarifications.

Thank you for contributing! 🚀
