# litemoe — unified entry point (mirrors the root reframework Makefile style).
#
#   make install    pip install -e .
#   make test       run the test suite (pytest, tests/)
#   make run        one-shot generation: make run CONFIG=configs/tiny.yaml
#   make clean      remove build artifacts and caches

PYTHON ?= python3
CONFIG ?= configs/tiny.yaml

.PHONY: help install test run clean

help:
	@echo "litemoe command entry point"
	@echo ""
	@echo "  make install    pip install -e ."
	@echo "  make test       run the test suite"
	@echo "  make run        one-shot generation (CONFIG=configs/tiny.yaml)"
	@echo "  make clean      remove build artifacts + caches"

install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) -m litemoe.cli run --config $(CONFIG)

clean:
	rm -rf build/ dist/ *.egg-info litemoe.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
