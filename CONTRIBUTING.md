# Contributing to Nüm Agents SDK

Thank you for your interest in contributing to the Nüm Agents SDK! This document provides guidelines and instructions for contributing to the project.

**IMPORTANT NOTICE**: This is a proprietary project owned by Lionel TAGNE. All contributions you make will be under the proprietary license terms outlined in the LICENSE file. By contributing to this project, you agree to assign all rights to your contributions to the project owner.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to maintain a positive and inclusive community.

## Getting Started

1. **Fork the repository**: Start by forking the [Nüm Agents SDK repository](https://github.com/numtema/num-agents-sdk).

2. **Clone your fork**: Clone your fork to your local machine.

   ```bash
   git clone https://github.com/your-username/num-agents-sdk.git
   cd num-agents-sdk
   ```


3. **Set up the development environment**: Install the development dependencies.

   ```bash
   pip install -e ".[dev]"
   ```


4. **Create a branch**: Create a branch for your contribution.

   ```bash
   git checkout -b feature/your-feature-name
   ```


## Development Workflow

1. **Make your changes**: Implement your feature or fix the bug.

2. **Write tests**: Add tests for your changes to ensure they work as expected.

3. **Run the tests**: Make sure all tests pass.

   ```bash
   pytest
   ```


4. **Format your code**: Ensure your code follows our style guidelines.

   ```bash
   black .
   isort .
   ```


5. **Check for linting errors**: Run the linter to check for any issues.

   ```bash
   flake8
   ```


6. **Commit your changes**: Write a clear commit message.

   ```bash
   git commit -m "Add feature: your feature description"
   ```


7. **Push to your fork**: Push your changes to your fork.

   ```bash
   git push origin feature/your-feature-name
   ```


8. **Create a pull request**: Submit a pull request from your branch to the main repository.

## Pull Request Guidelines

1. **Describe your changes**: Provide a clear description of what your pull request does.

2. **Reference issues**: If your pull request addresses an issue, reference it in the description.

3. **Keep it focused**: Each pull request should address a single concern.

4. **Update documentation**: If your changes affect the user experience, update the documentation accordingly.

5. **Add tests**: Include tests for new features or bug fixes.

## Adding New Features

When adding new features to the Nüm Agents SDK, consider the following:

1. **Universe-Based Architecture**: If your feature is a new module, decide which universe it belongs to or if it requires a new universe.

2. **Backward Compatibility**: Ensure your changes don't break existing functionality.

3. **Documentation**: Add documentation for your feature in the appropriate files.

4. **Examples**: Consider adding examples to demonstrate the use of your feature.

## Extending the Universe Catalog

To add a new universe to the catalog:

1. **Define the universe**: Add a new entry to the universe catalog in `config/univers_catalog.yaml`.

2. **Implement the modules**: Create the necessary module classes in the appropriate directories.

3. **Update documentation**: Add documentation for your universe in `docs/univers_catalog.md`.

4. **Add tests**: Write tests for your new modules.

## Reporting Issues

If you find a bug or have a suggestion for improvement:

1. **Check existing issues**: Make sure the issue hasn't already been reported.

2. **Create a new issue**: If it's a new issue, create a new issue with a clear description.

3. **Include details**: Provide as much detail as possible, including steps to reproduce, expected behavior, and actual behavior.

## Code Style

We follow these style guidelines:

1. **PEP 8**: Follow the [PEP 8](https://pep8.org/) style guide for Python code.

2. **Docstrings**: Use Google-style docstrings for all functions, classes, and methods.

3. **Type Hints**: Include type hints for function parameters and return values.

4. **Imports**: Organize imports using `isort`.

5. **Formatting**: Use `black` for code formatting.

## License

By contributing to the Nüm Agents SDK, you agree that your contributions will be licensed under the project's [Numtema](LICENSE).

## Questions?

If you have any questions about contributing, please reach out to the maintainers or open an issue for discussion.

Thank you for contributing to the Nüm Agents SDK!
