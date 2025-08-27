MAKEFLAGS += --no-print-directory

## ---------------------------------------------------------------------------
## | Redis RAG Workbench                                                     |
## | Dev Commands                                                            |
## ---------------------------------------------------------------------------

help:              ## Show this help
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)


.PHONY: install
install:           ## Install/sync dependencies
	@uv sync --all-extras


.PHONY: setup
setup:             ## Initial project setup (install deps and copy env)
	@echo "🚀 Setting up Redis RAG Workbench..."
	@$(MAKE) install
	@if [ ! -f .env ]; then cp .env-example .env && echo "📝 Created .env from .env-example - please edit with your credentials"; fi


.PHONY: dev
dev:               ## Run development server with hot reload
	@echo "🔧 Starting development server..."
	@$(MAKE) install
	@uv run fastapi dev main.py --host 0.0.0.0 --port 8000


.PHONY: serve
serve:             ## Run production server
	@echo "🚀 Starting production server..."
	@$(MAKE) install
	@uv run fastapi run main.py --host 0.0.0.0 --port 8000


.PHONY: format
format:            ## Format and lint code
	@echo "🎨 Formatting code..."
	@$(MAKE) install
	@uv run ruff check --fix
	@uv run ruff format


.PHONY: check
check:             ## Run code quality checks without fixing
	@echo "🔍 Running code quality checks..."
	@$(MAKE) install
	@uv run ruff check
	@uv run ruff format --check


## ---------------------------------------------------------------------------
## | Docker Commands                                                            |
## ---------------------------------------------------------------------------
.PHONY: docker
docker:            ## Rebuild and run docker container
	@echo "🐳 Rebuilding and starting Docker services..."
	@docker compose -f docker/compose.yml down
	@docker compose -f docker/compose.yml up -d --build


.PHONY: docker-up
docker-up:         ## Start all services with Docker Compose
	@echo "🐳 Starting Docker services..."
	@docker compose -f docker/compose.yml up -d


.PHONY: docker-logs
docker-logs:       ## View Docker logs
	@docker compose -f docker/compose.yml logs -f app


.PHONY: docker-down
docker-down:       ## Stop Docker services
	@docker compose -f docker/compose.yml down


.PHONY: clean
clean:             ## Clean build artifacts and caches
	@echo "🧹 Cleaning up..."
	@rm -rf .mypy_cache/ .ruff_cache/ __pycache__/
	@find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
