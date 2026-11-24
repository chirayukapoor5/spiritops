UV ?= uv
COMPOSE ?= docker compose
MODE ?= demo

.PHONY: setup data train up test demo benchmark down
setup:
	$(UV) sync --locked --python 3.11

data:
	$(UV) run --locked spiritops data --mode $(MODE)

train:
	$(UV) run --locked spiritops train

up:
	$(COMPOSE) up --build --detach --wait --wait-timeout 1800

test:
	$(UV) run --locked ruff check .
	$(UV) run --locked ruff format --check .
	$(UV) run --locked mypy src dashboard scripts
	$(UV) run --locked pytest tests/unit
	$(UV) run --locked pytest tests/integration tests/end_to_end

demo:
	$(UV) run --locked spiritops demo --mode $(MODE)

benchmark:
	$(UV) run --locked spiritops benchmark

down:
	$(COMPOSE) down
