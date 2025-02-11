FROM python:3.11-slim

ENV POETRY_VERSION=2.0

RUN pip install "poetry==${POETRY_VERSION}"

# Copy only requirements to cache them in docker layer
WORKDIR /app
COPY poetry.lock pyproject.toml /app/

# Project initialization:
RUN poetry install --no-interaction --no-ansi --no-root

# Creating folders, and files for a project:
COPY . .

ENV PORT 8000
EXPOSE ${PORT}

CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--reload"]

