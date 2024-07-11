# Redis RAG Workbench

Redis RAG Workbench is a playground for exploring Retrieval-Augmented Generation (RAG) techniques using Redis. This project provides a collection of demos showcasing various RAG implementations and utilities.

## Features

- Integration with Redis for vector storage and caching
- Support for various LLM models and reranking techniques
- Modular architecture for easy extension and customization (soon)

## Prerequisites

- Python 3.11 or higher
- Redis server
- OpenAI API key
- Cohere API key (for certain reranking features)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/redis-rag-workbench.git
   cd redis-rag-workbench
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install Poetry (see detailed instructions at [Poetry Installation](https://python-poetry.org/docs/#installation)):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Install the required dependencies:
   ```bash
   poetry install
   ```

4. Set up your environment variables by creating a `.env` file in the project root:
   ```env
   REDIS_URL=your_redis_url
   OPENAI_API_KEY=your_openai_api_key
   COHERE_API_KEY=your_cohere_api_key
   ```

## Running the Application

To start the application, run:

```bash
uvicorn main:app --reload
```

This will start the server, and you can access the demos by navigating to `http://localhost:8000` in your web browser.

## Project Structure

- `main.py`: The entry point of the application
- `demos/`: Contains individual RAG demo implementations
- `shared_components/`: Reusable utilities and components
- `static/`: Static assets for the web interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.