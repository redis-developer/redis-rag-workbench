from enum import StrEnum


class LLMs(StrEnum):
    openai = "openai"
    azure = "azure-openai"
    vertexai = "vertexai"

def openai_models():
    return [
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo-instruct",
        "text-ada-001",
        "ada",
        "text-babbage-001",
        "babbage",
        "text-curie-001",
        "curie",
        "davinci",
        "text-davinci-003",
        "text-davinci-002",
        "code-davinci-002",
        "code-davinci-001",
        "code-cushman-002",
        "code-cushman-001",
    ]

def gemini_models():
    return [
        "gemini-2.0-flash-thinking-exp-01-21",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
    ]

def vertex_embedding_models():
    return [
        "text-embedding-004", 
        "textembedding-gecko@003", 
        "textembedding-gecko@001"
    ]

# Pricing info: https://cloud.google.com/vertex-ai/generative-ai/pricing
def calculate_vertexai_cost(input_tokens, output_tokens, model):
    match model:
        case "gemini-2.0-flash-thinking-exp-01-21":
            return ((input_tokens / 1_000_000) * 0.15) + ((output_tokens / 1_000_000) * 0.6)
        case "gemini-1.5-flash":
            return (((input_tokens * 4) / 1000) * 0.00001875) + (((output_tokens * 4) / 1000) * 0.000075)
        case "gemini-1.5-pro":
            return (((input_tokens * 4) / 1000) * 0.0003125) + (((output_tokens * 4) / 1000) * 0.00125)
        case "gemini-1.0-pro":
            return (((input_tokens * 4) / 1000) * 0.000125) + (((output_tokens * 4) / 1000) * 0.000375)