MAKEFLAGS += --no-print-directory

## ---------------------------------------------------------------------------
## | The purpose of this Makefile is to provide all the functionality needed |
## | to install, build, run, and deploy the RAG Workbench.                   |
## ---------------------------------------------------------------------------

help:              ## Show this help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

install:           ## Install dependencies
	@uv sync --all-extras

dev:               ## Run a dev server
	@$(MAKE) install
	@fastapi dev main.py

serve:             ## Run a production server
	@$(MAKE) install
	@fastapi run main.py

docker:            ## Rebuild and run docker container
	@docker compose down
	@$(MAKE) docker-prune
	@docker compose up -d

docker-prune:      ## Prune unused docker images, volumes, and builder cache
	@docker image prune -a -f
	@docker volume prune -a -f
	@docker builder prune -a -f

format:            ## Format code
	@$(MAKE) install
	@ruff check --fix
	@ruff format

lock: install
	@uv lock

clean:             ## Clean build files
	@rm -rf .venv/ .mypy_cache/ .ruff_cache/ __pycache__/
	@find . -type d -name __pycache__ -exec rm -r {} \+
