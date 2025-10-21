# Development Guide - Nüm Agents SDK

This guide covers the development workflow, tools, and best practices for contributing to the Nüm Agents SDK.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Quality Tools](#code-quality-tools)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Release Process](#release-process)
- [Best Practices](#best-practices)

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Make (optional but recommended)

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/Creativityliberty/numagents.git
cd numagents

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
make install-dev
# Or manually:
# pip install -e ".[dev]"
# pre-commit install
```

This will:
- Install the package in editable mode
- Install all development dependencies
- Set up pre-commit hooks

---

## Development Environment

### Project Structure

```
numagents/
├── .github/
│   └── workflows/          # CI/CD workflows
│       ├── ci.yml          # Main CI pipeline
│       └── release.yml     # Release automation
├── docs/                   # Documentation
├── num_agents/             # Main package
│   ├── core.py            # Core classes
│   ├── exceptions.py      # Custom exceptions
│   ├── logging_config.py  # Logging setup
│   ├── serialization.py   # Flow serialization
│   ├── cli.py             # CLI commands
│   ├── composer/          # Agent generation
│   ├── graph/             # Graph generation
│   ├── orchestrator/      # Meta-orchestration
│   ├── univers/           # Universe catalog
│   └── utils/             # Utilities
├── tests/                  # Test suite
├── config/                 # Configuration files
├── examples/               # Example agents
├── .editorconfig          # Editor configuration
├── .gitignore             # Git ignore rules
├── .pre-commit-config.yaml # Pre-commit hooks config
├── Makefile               # Development commands
├── pyproject.toml         # Project configuration
└── README.md              # Main documentation
```

### Editor Configuration

The project includes a `.editorconfig` file that ensures consistent code style across different editors. Make sure your editor supports EditorConfig.

**Recommended VS Code Extensions:**
- Python
- Pylance
- Black Formatter
- isort
- EditorConfig for VS Code
- markdownlint

---

## Code Quality Tools

We use multiple tools to maintain high code quality:

### 1. Black - Code Formatting

Black automatically formats Python code to a consistent style.

```bash
# Check formatting
make lint  # or: black --check .

# Auto-format
make format  # or: black .
```

**Configuration** (`pyproject.toml`):
```toml
[tool.black]
line-length = 100
```

### 2. isort - Import Sorting

Isort organizes imports alphabetically and separates them into sections.

```bash
# Check imports
isort --check-only .

# Auto-sort imports
make format  # or: isort .
```

**Configuration** (`pyproject.toml`):
```toml
[tool.isort]
profile = "black"
line_length = 100
```

### 3. Ruff - Fast Python Linter

Ruff is an extremely fast Python linter written in Rust.

```bash
# Run ruff
make lint  # or: ruff check .

# Auto-fix issues
ruff check --fix .
```

**Configuration** (`pyproject.toml`):
```toml
[tool.ruff]
line-length = 100
select = ["E", "F", "B"]
```

### 4. mypy - Type Checking

Mypy performs static type checking to catch type-related bugs.

```bash
# Run type checking
make type-check  # or: mypy num_agents
```

**Configuration** (`pyproject.toml`):
```toml
[tool.mypy]
python_version = "3.8"
warn_return_any = true
disallow_untyped_defs = true
```

### 5. Bandit - Security Scanner

Bandit finds common security issues in Python code.

```bash
# Run security scan
bandit -r num_agents -c pyproject.toml
```

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit:

```bash
# Run all hooks manually
make pre-commit

# Skip hooks (not recommended)
git commit --no-verify

# Update hooks
make pre-commit-update
```

**Hooks enabled:**
- Trailing whitespace removal
- End-of-file fixer
- YAML/JSON/TOML validation
- Large file check
- Black formatting
- isort import sorting
- Ruff linting
- mypy type checking
- Bandit security scan

---

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run in parallel (faster)
make test-parallel

# Run specific test file
pytest tests/test_core_improvements.py

# Run specific test
pytest tests/test_core_improvements.py::TestFlowImprovements::test_flow_with_name
```

### Test Structure

Tests are organized by module:

```
tests/
├── test_core_improvements.py     # Core functionality tests
├── test_serialization.py         # Serialization tests
├── test_composer.py              # Agent generation tests
├── test_logical_graph.py         # Graph generation tests
├── test_cli_integration.py       # CLI tests
└── test_univers_catalog_loader.py # Catalog tests
```

### Writing Tests

Follow these guidelines when writing tests:

1. **Use descriptive names:**
   ```python
   def test_flow_with_initial_data() -> None:
       """Test flow execution with initial data injection."""
   ```

2. **Follow AAA pattern:**
   ```python
   def test_example() -> None:
       # Arrange
       flow = Flow(name="Test")

       # Act
       results = flow.execute()

       # Assert
       assert results is not None
   ```

3. **Use pytest fixtures for common setup:**
   ```python
   @pytest.fixture
   def sample_flow() -> Flow:
       return Flow(name="SampleFlow")

   def test_with_fixture(sample_flow: Flow) -> None:
       assert sample_flow.name == "SampleFlow"
   ```

4. **Test edge cases and errors:**
   ```python
   def test_missing_start_node() -> None:
       flow = Flow()
       with pytest.raises(FlowConfigurationError):
           flow.validate()
   ```

### Coverage Requirements

- **Minimum coverage:** 80%
- **Target coverage:** 90%+

```bash
# Generate coverage report
make test-cov

# Open HTML coverage report
make coverage-report
```

Coverage reports are generated in:
- Terminal: Inline during test run
- HTML: `htmlcov/index.html`
- XML: `coverage.xml` (for CI/CD)

---

## CI/CD Pipeline

### GitHub Actions Workflows

#### Main CI Workflow (`.github/workflows/ci.yml`)

Runs on:
- Push to `main`, `develop`, `claude/**` branches
- Pull requests to `main`, `develop`

**Jobs:**
1. **lint**: Code formatting and linting checks
2. **type-check**: mypy type checking
3. **test**: Test on Python 3.8, 3.9, 3.10, 3.11
4. **test-cli**: CLI command testing
5. **build**: Package building and validation

**Coverage:**
- Uploaded to Codecov from Python 3.11 tests
- Badge displayed in README

#### Release Workflow (`.github/workflows/release.yml`)

Triggers on:
- Tags matching `v*.*.*` (e.g., `v0.1.0`)

**Actions:**
- Builds distribution packages
- Creates GitHub release with artifacts
- Generates release notes

### Local CI Simulation

Run all CI checks locally before pushing:

```bash
# Run all checks
make check

# Or run individually
make lint
make type-check
make test-cov
make build-check
```

---

## Release Process

### Version Bumping

1. Update version in `num_agents/__init__.py`:
   ```python
   __version__ = "0.2.0"
   ```

2. Update version in `pyproject.toml`:
   ```toml
   version = "0.2.0"
   ```

3. Update `CHANGELOG.md` (if exists) with release notes

### Creating a Release

```bash
# Ensure all tests pass
make check

# Commit version changes
git add num_agents/__init__.py pyproject.toml
git commit -m "Bump version to 0.2.0"

# Create and push tag
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0

# GitHub Actions will automatically:
# - Run all CI checks
# - Build packages
# - Create GitHub release
```

### Build Package Locally

```bash
# Build distribution packages
make build

# Check package with twine
make build-check

# Inspect built packages
ls -lh dist/
```

---

## Best Practices

### Code Style

1. **Follow PEP 8** with these modifications:
   - Line length: 100 characters
   - Use double quotes for strings

2. **Use type hints everywhere:**
   ```python
   def process_data(data: List[str]) -> Dict[str, Any]:
       ...
   ```

3. **Write descriptive docstrings:**
   ```python
   def my_function(param: str) -> bool:
       """
       Brief description of what the function does.

       Args:
           param: Description of parameter

       Returns:
           Description of return value

       Raises:
           ValueError: When param is invalid
       """
   ```

### Git Workflow

1. **Branch naming:**
   - Features: `feature/description`
   - Bugs: `fix/description`
   - Refactoring: `refactor/description`

2. **Commit messages:**
   ```
   Type: Brief description (max 72 chars)

   Detailed explanation if needed.

   - Bullet points for changes
   - Another change

   Fixes #123
   ```

3. **Pull requests:**
   - Create descriptive PR titles
   - Fill out PR template
   - Link related issues
   - Ensure all CI checks pass

### Error Handling

1. **Use custom exceptions:**
   ```python
   from num_agents.exceptions import FlowExecutionError

   if not self._start_node:
       raise FlowConfigurationError("No start node defined")
   ```

2. **Provide context in exceptions:**
   ```python
   raise NodeExecutionError(
       "Node execution failed",
       node_name=self.name,
       node_id=self.id,
       details={"error": str(e)}
   )
   ```

### Logging

1. **Use structured logging:**
   ```python
   from num_agents.logging_config import get_logger

   logger = get_logger(__name__)
   logger.info(f"Processing {count} items")
   ```

2. **Log levels:**
   - `DEBUG`: Detailed information for debugging
   - `INFO`: General informational messages
   - `WARNING`: Warning messages (recoverable errors)
   - `ERROR`: Error messages (execution failures)

### Performance

1. **Minimize logging overhead:**
   - Logging is disabled by default
   - Enable only when needed: `enable_logging=True`

2. **Use hooks efficiently:**
   - Hooks add ~1-2ms overhead
   - Only add necessary hooks

3. **Avoid cycles in flows:**
   - Flow validation detects cycles
   - Design acyclic graphs

---

## Troubleshooting

### Pre-commit Hook Failures

```bash
# If hooks fail, they often auto-fix issues
# Stage the fixes and commit again
git add .
git commit

# Skip hooks temporarily (not recommended)
git commit --no-verify
```

### Test Failures

```bash
# Run single test for debugging
pytest tests/test_file.py::test_name -v

# Run with debugging output
pytest tests/test_file.py -vv -s

# Use pytest debugger
pytest --pdb
```

### Type Checking Errors

```bash
# Run mypy with detailed output
mypy num_agents --show-error-codes

# Ignore specific error (last resort)
result = some_function()  # type: ignore[error-code]
```

---

## Getting Help

- **Documentation:** Check `docs/` folder
- **Issues:** Open a GitHub issue
- **Discussions:** Use GitHub Discussions

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.

---

*Happy coding! 🚀*
