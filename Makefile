# Makefile for syntx
# Portable Python discovery: VIRTUAL_ENV -> CONDA_PREFIX -> python3 on PATH
ifdef VIRTUAL_ENV
    DEFAULT_PYTHON := $(VIRTUAL_ENV)/bin/python
else ifdef CONDA_PREFIX
    DEFAULT_PYTHON := $(CONDA_PREFIX)/bin/python
else
    DEFAULT_PYTHON := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python3)
endif
PYTHON ?= $(DEFAULT_PYTHON)
PIP    ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF   ?= $(PYTHON) -m ruff
PKG    := src/syntx

.PHONY: help install install-dev install-all test test-fast test-slow test-all test-cov test-strict lint format check compile audit clean build release pypi

help:
	@echo "syntx Development & Maintenance Tasks"
	@echo "====================================="
	@echo "  make install      Editable install of syntx"
	@echo "  make install-dev  Editable install with [test] extras"
	@echo "  make install-all  Alias for install-dev"
	@echo "  make test         Run fast unit tests (not slow)"
	@echo "  make test-fast    Alias for test"
	@echo "  make test-slow    Run slow 3D / benchmark tests"
	@echo "  make test-all     Run full test suite including slow tests"
	@echo "  make test-cov     Run fast unit tests with coverage"
	@echo "  make test-strict  Run full test suite including slow tests"
	@echo "  make lint         Run ruff check and antsxqc.lint"
	@echo "  make format       Run ruff format and check --fix"
	@echo "  make compile      Byte-compile package sources without SyntaxWarnings"
	@echo "  make audit        Run compile + lint + test"
	@echo "  make check        Alias for audit"
	@echo "  make build        Build sdist and wheel"
	@echo "  make clean        Remove caches and build artifacts"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[test]"

install-all: install-dev

test:
	CUDA_VISIBLE_DEVICES="" PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTEST) -m "not slow" -q

test-fast: test

test-slow:
	CUDA_VISIBLE_DEVICES="" PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTEST) -m "slow" -v

test-all: test-strict

test-cov:
	CUDA_VISIBLE_DEVICES="" PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTEST) -m "not slow" --cov=syntx --cov-report=term-missing

test-strict:
	CUDA_VISIBLE_DEVICES="" PYTORCH_ENABLE_MPS_FALLBACK=1 $(PYTEST) -v

lint:
	$(RUFF) check $(PKG) tests
	$(PYTHON) -m antsxqc.lint $(PKG)

format:
	$(RUFF) format $(PKG) tests
	$(RUFF) check --fix $(PKG) tests

compile:
	$(PYTHON) -W error::SyntaxWarning -m compileall $(PKG) tests

audit: compile lint test

check: audit

build: clean
	$(PYTHON) -m build

release: clean build
	$(PYTHON) -m twine upload dist/*

pypi: clean build
	$(PYTHON) -m twine upload dist/*
	@VERSION=$$($(PYTHON) -c "import syntx; print(syntx.__version__)"); \
	git tag v$$VERSION && git push origin v$$VERSION

clean:
	rm -rf build dist src/*.egg-info *.egg-info .eggs .pytest_cache .coverage* htmlcov .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
