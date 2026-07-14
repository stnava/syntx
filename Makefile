# Makefile for syntx

# Detect active virtual env or default to system/active environment
ifdef VIRTUAL_ENV
    VENV_BIN = $(VIRTUAL_ENV)/bin
    PYTHON = $(VENV_BIN)/python
    PIP = $(VENV_BIN)/pip
    PYTEST = $(VENV_BIN)/pytest
else
    # Fallback to current path executables (which will be the active venv if activated in the shell)
    PYTHON = python3
    PIP = pip
    PYTEST = pytest
endif

.PHONY: install test test-all clean release

install:
	@echo "Installing syntx in editable mode using: $(PYTHON)"
	$(PIP) install -e .

test:
	@echo "Running test suite in FAST mode with coverage using: $(PYTHON)"
	$(PYTEST) --cov=syntx --cov-report=term-missing

test-all:
	@echo "Running FULL test suite (including slow tests) with coverage using: $(PYTHON)"
	$(PYTEST) --runslow --cov=syntx --cov-report=term-missing

clean:
	@echo "Cleaning build artifacts and cache files..."
	rm -rf build/ dist/ src/syntx.egg-info/ syntx.egg-info/ .eggs/
	rm -f *.nii.gz *.mat
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +

release: clean
	@echo "Building distribution packages (sdist and wheel) using: $(PYTHON)"
	$(PYTHON) -m build
	@echo "Uploading packages to PyPI via twine..."
	$(PYTHON) -m twine upload dist/*
