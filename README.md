<div align="center">
<div><img src="assets/redis-logo.svg" style="width: 130px"> </div>
<h1>RAG Workbench</h1>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Language](https://img.shields.io/github/languages/top/redis-developer/redis-rag-workbench)
![GitHub last commit](https://img.shields.io/github/last-commit/redis-developer/redis-rag-workbench)

</div>

🛠️ **Redis RAG Workbench** is a development playground for exploring Retrieval-Augmented Generation (RAG) techniques with Redis. Upload a PDF and begin building a RAG app to chat with the document, taking full advantage of Redis features like **vector search**, **semantic caching**, **LLM memory**, and **semantic routing**.

<div></div>


## Prerequisites

1. Make sure you have the following tools available:
   - [Docker](https://www.docker.com/products/docker-desktop/)
   - Python >= 3.11 and [Poetry](https://python-poetry.org/docs/#installation)
   - [OpenAI API key](https://platform.openai.com/)
   - [Cohere API key](https://cohere.com/) (for optional reranking features)


2. Clone the repository:
   ```bash
   git clone https://github.com/redis-developer/redis-rag-workbench.git
   cd redis-rag-workbench
   ```


3. Set up your environment variables by creating a `.env` file in the project root:
   ```env
   REDIS_URL=your_redis_url
   OPENAI_API_KEY=your_openai_api_key
   COHERE_API_KEY=your_cohere_api_key
   ```


## Start the workbench
In the root of the repository, run the following to spin up the docker compose stack:
   ```bash
   docker compose -f docker-compose.yml up
   ```

> This will start the server, and you can access the workbench by navigating to `http://localhost:8000` in your web browser.

<div><img src="assets/workbench_sample.png" style="width: 625px"> </div>

> The first time the application runs, it will have to download model weights from huggingface and may take a few minutes.


## Project Structure

- `main.py`: The entry point of the application
- `demos/`: Contains workbench demo implementation
- `shared_components/`: Reusable utilities and components
- `static/`: Static assets for the web interface

## Contributing

🤝 Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.