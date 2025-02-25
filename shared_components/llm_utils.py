from enum import StrEnum


class LLMs(StrEnum):
    openai = "openai"
    azure = "azure-openai"
    vertexai = "vertexai"


def sort_models(models_dlist):
    return sorted(models_dlist, key=lambda d: d["model"])


oai_models = sort_models(
    [
        {"model": "gpt-4o", "default": False},
        {"model": "gpt-4o-2024-05-13", "default": False},
        {"model": "gpt-4", "default": False},
        {"model": "gpt-4-0314", "default": False},
        {"model": "gpt-4-0613", "default": False},
        {"model": "gpt-4-32k", "default": False},
        {"model": "gpt-4-32k-0314", "default": False},
        {"model": "gpt-4-32k-0613", "default": False},
        {"model": "gpt-3.5-turbo", "default": True},
        {"model": "gpt-3.5-turbo-0301", "default": False},
        {"model": "gpt-3.5-turbo-0613", "default": False},
        {"model": "gpt-3.5-turbo-16k", "default": False},
        {"model": "gpt-3.5-turbo-16k-0613", "default": False},
        {"model": "gpt-3.5-turbo-instruct", "default": False},
        {"model": "text-ada-001", "default": False},
        {"model": "ada", "default": False},
        {"model": "text-babbage-001", "default": False},
        {"model": "babbage", "default": False},
        {"model": "text-curie-001", "default": False},
        {"model": "curie", "default": False},
        {"model": "davinci", "default": False},
        {"model": "text-davinci-003", "default": False},
        {"model": "text-davinci-002", "default": False},
        {"model": "code-davinci-002", "default": False},
        {"model": "code-davinci-001", "default": False},
        {"model": "code-cushman-002", "default": False},
        {"model": "code-cushman-001", "default": False},
    ]
)

oai_embedding_models = sort_models(
    [
        {"model": "text-embedding-ada-002", "default": True},
        {"model": "text-embedding-3-small", "default": False},
    ]
)

vai_models = sort_models(
    [
        {"model": "gemini-2.0-flash-thinking-exp-01-21", "default": True},
        {"model": "gemini-1.5-flash", "default": False},
        {"model": "gemini-1.5-pro", "default": False},
        {"model": "gemini-1.0-pro", "default": False},
    ]
)

vai_embedding_models = sort_models(
    [
        {"model": "text-embedding-004", "default": True},
        {"model": "textembedding-gecko@003", "default": False},
        {"model": "textembedding-gecko@001", "default": False},
    ]
)


def get_models(models_dlist):
    return [v["model"] for v in models_dlist]


def get_default(models_dlist):
    return next(v["model"] for v in models_dlist if v["default"])


def openai_models():
    return get_models(oai_models)


def default_openai_model():
    return get_default(oai_models)


def openai_embedding_models():
    return get_models(oai_embedding_models)


def default_openai_embedding_model():
    return get_default(oai_embedding_models)


def vertex_models():
    return get_models(vai_models)


def default_vertex_model():
    return get_default(vai_models)


def vertex_embedding_models():
    return get_models(vai_embedding_models)


def default_vertex_embedding_model():
    return get_default(vai_embedding_models)


# Pricing info: https://cloud.google.com/vertex-ai/generative-ai/pricing
def calculate_vertexai_cost(input_tokens, output_tokens, model):
    match model:
        case "gemini-2.0-flash-thinking-exp-01-21":
            return ((input_tokens / 1_000_000) * 0.15) + (
                (output_tokens / 1_000_000) * 0.6
            )
        case "gemini-1.5-flash":
            return (((input_tokens * 4) / 1000) * 0.00001875) + (
                ((output_tokens * 4) / 1000) * 0.000075
            )
        case "gemini-1.5-pro":
            return (((input_tokens * 4) / 1000) * 0.0003125) + (
                ((output_tokens * 4) / 1000) * 0.00125
            )
        case "gemini-1.0-pro":
            return (((input_tokens * 4) / 1000) * 0.000125) + (
                ((output_tokens * 4) / 1000) * 0.000375
            )
