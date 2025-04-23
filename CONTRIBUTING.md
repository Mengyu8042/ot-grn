# Contributing to ot-grn

We welcome contributions to **ot-grn**, whether it's bug reports, feature requests, or pull requests.

## How to Contribute

### 1. Fork the Repository
Click the **Fork** button on the top right of the GitHub page and clone your fork locally:

```bash
git clone https://github.com/your-username/ot-grn.git
cd ot-grn
```

### 2. Set Up the Development Environment

We recommend creating a virtual environment:

```bash
pip install -e .[dev]
```

This installs the package in editable mode along with development dependencies, such as `pytest`, `flake8`, and `black`.

### 3. Run Tests

Before submitting a pull request, ensure all tests pass:

```bash
pytest
```

### 4. Open a Pull Request

Push your changes to a new branch and open a pull request to the `main` branch. Please clearly describe your changes.


## Code Style

We follow standard Python [PEP8](https://pep8.org/) conventions. You may run `flake8` or `black` for formatting if preferred.

## Reporting Bugs or Requesting Features

Please open an issue if you:

- Encounter unexpected behavior
- Want to propose a new feature
- Need help understanding the code or results


Thank you for helping improve **ot-grn**!