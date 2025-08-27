<div align="center">
<div><img src="assets/redis-logo.svg" style="width: 130px"> </div>
<h1>🚀 Redis RAG Workbench</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Language](https://img.shields.io/github/languages/top/redis-developer/redis-rag-workbench)
![GitHub last commit](https://img.shields.io/github/last-commit/redis-developer/redis-rag-workbench)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green)

**🎯 The ultimate RAG development playground powered by Redis**

</div>

🔥 **Redis RAG Workbench** is your go-to development environment for building and experimenting with **Retrieval-Augmented Generation (RAG)** applications. Drop in a PDF, chat with your documents, and harness the full power of Redis for **lightning-fast vector search**, **intelligent semantic caching**, **persistent LLM memory**, and **smart semantic routing**.

✨ **What makes this special?**
- 🚀 **One-command setup** - Get started in seconds with `make setup`
- ⚡ **Multi-LLM support** - OpenAI, Azure OpenAI, Google VertexAI
- 🎯 **Redis-powered** - Vector search, caching, and memory management
- 🐳 **Docker ready** - Consistent development across all environments
- 🔧 **Developer-first** - Hot reload, code formatting, and quality checks built-in

---

## Table of Contents

- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [Available Commands](#available-commands)
  - [Development Workflows](#development-workflows)
  - [Environment Configuration](#environment-configuration)
- [Using Google VertexAI](#using-google-vertexai)
- [Project Structure](#project-structure)
- [Connecting to Redis Cloud](#connecting-to-redis-cloud)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Learn More](#learn-more)


## Quick Start

**Get up and running in 3 commands:**

```bash
git clone https://github.com/redis-developer/redis-rag-workbench.git
cd redis-rag-workbench
make setup && make dev
```

Then visit `http://localhost:8000` and start chatting with your PDFs! 🎉

---

## Prerequisites

1. Make sure you have the following tools available:
   - [Docker](https://www.docker.com/products/docker-desktop/)
   - [uv](https://docs.astral.sh/uv/)
   - [make](https://www.make.com/en)
2. Setup one or more of the following:
   - [OpenAI API](https://platform.openai.com/)
     - You will need an API Key
   - [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
     - You will need an API Key
   - [Google VertexAI](https://cloud.google.com/vertex-ai?hl=en)
     - [Using Google VertexAI](#using-google-vertexai)
3. Get a [Cohere API key](https://cohere.com/) (for optional reranking features)


## Getting Started

> 🌐 Access the workbench at `http://localhost:8000` in your web browser.

<div><img src="assets/workbench_sample.png" style="width: 625px"> </div>

> ⏱️ First run may take a few minutes to download model weights from Hugging Face.

### Available Commands

| Command | Description |
|---------|-------------|
| `make setup` | Initial project setup (install deps & create .env) |
| `make install` | Install/sync dependencies |
| `make dev` | Start development server with hot reload |
| `make serve` | Start production server |
| `make format` | Format and lint code |
| `make check` | Run code quality checks without fixing |
| `make docker` | Rebuild and run Docker containers |
| `make docker-up` | Start Docker services (without rebuild) |
| `make docker-logs` | View Docker application logs |
| `make docker-down` | Stop Docker services |
| `make clean` | Clean build artifacts and caches |

### Development Workflows

**Local Development:**
```bash
make setup           # One-time setup
# Edit .env with your API keys
make dev             # Start development server
```

**Docker Development:**
```bash
make setup           # One-time setup  
# Edit .env with your API keys
make docker          # Build and start containers
make docker-logs     # View logs
```

**Docker Management:**
```bash
make docker-up       # Start existing containers
make docker-down     # Stop all services
make docker-logs     # Follow application logs
```

**Code Quality:**
```bash
make format          # Auto-fix formatting issues
make check           # Check code quality without changes
```

### Environment Configuration

The project uses a single `.env` file for configuration. Copy from the example:

```bash
cp .env-example .env
```

Required variables:
- `REDIS_URL` - Redis connection (auto-configured for Docker)
- `COHERE_API_KEY` - For reranking features

At least one LLM provider:
- `OPENAI_API_KEY` - OpenAI API access
- `AZURE_OPENAI_*` - Azure OpenAI configuration  
- `GOOGLE_APPLICATION_CREDENTIALS` - Google VertexAI credentials

## Using Google VertexAI
The workbench can be used with VertexAI, but requires you to set up your credentials using the `gcloud` CLI. The easiest way to do this is as follows:

1. Make sure you have a [gcloud project setup](https://cloud.google.com/vertex-ai/docs/start/cloud-environment) with the VertexAI API enabled.
2. Install the [gcloud CLI](https://cloud.google.com/sdk/docs/install)
3. Follow the instructions to run the [`gcloud auth application-default login`](https://cloud.google.com/docs/authentication/application-default-credentials#personal) command
4. Copy the JSON from the generated `application_default_credentials.json` into your `.env` file using the `GOOGLE_APPLICATION_CREDENTIALS` variable
5. Set the `GOOGLE_CLOUD_PROJECT_ID` environment variable in your `.env` file to the associated gcloud project you want to use.

## Project Structure

- `main.py`: The entry point of the application
- `demos/`: Contains workbench demo implementation
- `shared_components/`: Reusable utilities and components
- `static/`: Static assets for the web interface

## Contributing

🤝 Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Connecting to Redis Cloud

If you don't yet have a database setup in Redis Cloud [get started here for free](https://redis.io/try-free/).

To connect to a Redis Cloud database, log into the console and find the following:

1. The `public endpoint` (looks like `redis-#####.c###.us-east-1-#.ec2.redns.redis-cloud.com:#####`)
1. Your `username` (`default` is the default username, otherwise find the one you setup)
1. Your `password` (either setup through Data Access Control, or available in the `Security` section of the database
   page.

Combine the above values into a connection string and put it in your `.env` file. It should look something like the following:

```bash
REDIS_URL="redis://default:<password>@redis-#####.c###.us-west-2-#.ec2.redns.redis-cloud.com:#####"
```

> 📝 **Note:** When using Docker, the Redis URL is automatically configured to use the internal Docker network. Your `.env` file can contain either a local Redis URL (`redis://localhost:6379`) or a Redis Cloud URL - both will work with Docker.

## Troubleshooting

### Apple Silicon (M1+)

If you find that `docker` will not work, it's possible you need to add the following line in the `docker/Dockerfile` (commented out in the Dockerfile for ease-of-use):

```dockerfile
RUN apt-get update && apt-get install -y build-essential
```

## Learn More

To learn more about Redis, take a look at the following resources:

- [Redis Documentation](https://redis.io/docs/latest/) - learn about Redis products, features, and commands.
- [Learn Redis](https://redis.io/learn/) - read tutorials, quick starts, and how-to guides for Redis.
- [Redis Demo Center](https://redis.io/demo-center/) - watch short, technical videos about Redis products and features.
