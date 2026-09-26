# Re (reframework) — unified command entry point.
#
# Every common action lives here so you don't have to remember the module paths:
#   make install      pip install the package in editable mode (+ core deps)
#   make server       add the FastAPI/uvicorn server deps too
#   make chat         interactive chat REPL (pass MODEL=/path/to/ckpt)
#   make run          one-shot generation from stdin (make run MODEL=... < prompt.txt)
#   make serve        start the OpenAI-compatible server
#   make test         run the test suite
#   make clean        remove build artifacts and caches

PYTHON      ?= python3
PKG         := reframework
MODEL       ?= .hf/tiny-llama
DEVICE      ?= auto
PORT        ?= 8000
HOST        ?= 127.0.0.1
EXTRA       ?=

.PHONY: help install install-server chat run serve test clean

help:
	@echo "Re command entry point"
	@echo ""
	@echo "  make install        install the package editable (+core deps)"
	@echo "  make server         install with the server extra (fastapi/uvicorn)"
	@echo "  make chat           chat REPL   (MODEL=/path/to/ckpt DEVICE=cpu|cuda|auto)"
	@echo "  make run            one-shot, prompt from stdin  (MODEL=... < prompt.txt)"
	@echo "  make serve          OpenAI-compatible server (HOST= PORT=)"
	@echo "  make test           run the test suite"
	@echo "  make clean          remove build artifacts + caches"

install:
	$(PYTHON) -m pip install -e .

server:
	$(PYTHON) -m pip install -e ".[server]"

chat:
	$(PYTHON) -m reframework.cli.chat $(MODEL) --device $(DEVICE) $(EXTRA)

run:
	$(PYTHON) -m reframework.cli.chat $(MODEL) --device $(DEVICE) --once $(EXTRA)

serve:
	$(PYTHON) -m reframework.server.app --host $(HOST) --port $(PORT) --model $(MODEL) --device $(DEVICE)

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf build/ dist/ *.egg-info $(PKG).egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
	find . -type d -name '.ruff_cache' -prune -exec rm -rf {} +
