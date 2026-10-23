PYTHON ?= /Users/stnava/venvs/ants/bin/python3
PYTEST ?= $(PYTHON) -m pytest
PIP    ?= $(PYTHON) -m pip
RUFF   ?= $(PYTHON) -m ruff
PKG    := src/syntx

.PHONY: help install install-dev test test-fast test-slow test-cov test-strict lint format check compile audit clean

help:
	@echo "Targets:"
	@echo "  install      Editable install of syntx"
	@echo "  install-dev  Editable install with [test] extras"
	@echo "  test         Run fast unit tests (not slow)"
	@echo "  test-fast    Alias for test"
	@echo "  test-slow    Run slow 3D / benchmark tests"
	@echo "  test-cov     Run fast unit tests with coverage"
	@echo "  test-strict  Run full test suite including slow tests"
	@echo "  lint         Run ruff check and antsxqc.lint"
	@echo "  format       Run ruff format"
	@echo "  compile      Byte-compile package sources"
	@echo "  audit        Run implementation standards and antsxqc lint checks"
	@echo "  check        lint + test"
	@echo "  clean        Remove caches and build artifacts"

install:
	$(PIP) install -e . --no-deps

install-dev:
	$(PIP) install -e ".[test]"

test:
	CUDA_VISIBLE_DEVICES="" PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTEST) -m "not slow" -q

test-fast: test

test-slow:
	CUDA_VISIBLE_DEVICES="" PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTEST) -m "slow" -v

test-cov:
	CUDA_VISIBLE_DEVICES="" PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTEST) -m "not slow" --cov=syntx --cov-report=term-missing

test-strict:
	CUDA_VISIBLE_DEVICES="" PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTEST) -v

lint:
	$(RUFF) check $(PKG) tests
	$(PYTHON) -m antsxqc.lint $(PKG)

format:
	$(RUFF) format $(PKG) tests

compile:
	$(PYTHON) -m py_compile $$(find $(PKG) -name '*.py')

audit:
	$(PYTHON) -m antsxqc.lint $(PKG)
	$(PYTEST) tests/test_implementation_standards.py tests/test_antsxqc_lint.py -q

check: lint test

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .coverage htmlcov .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
