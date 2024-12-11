import os
import os.path
from typing import Any, List, Optional

from datasets import Dataset
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import (
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    ChatOpenAI,
    OpenAIEmbeddings,
)
from langchain_redis import RedisChatMessageHistory, RedisVectorStore
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from ragas.llms import LangchainLLMWrapper
from redis.exceptions import ResponseError
from redisvl.extensions.llmcache import SemanticCache
from redisvl.extensions.router import SemanticRouter
from redisvl.utils.rerank import CohereReranker, HFCrossEncoderReranker
from ulid import ULID

from shared_components.cached_llm import CachedLLM
from shared_components.llm_utils import openai_models
from shared_components.pdf_manager import PDFManager, PDFMetadata
from shared_components.pdf_utils import process_file
from shared_components.converters import str_to_bool

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ChatApp:
    def __init__(self) -> None:
        self.session_id = None
        self.pdf_manager = None
        self.current_pdf_index = None

        self.redis_url = os.environ.get("REDIS_URL")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.cohere_api_key = os.environ.get("COHERE_API_KEY")

        self.azure_openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION")  # ex: 2024-08-01-preview
        self.azure_openai_api_key = os.environ.get("AZURE_OPENAI_API_KEY")  # ex: 1234567890abcdef
        self.azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")

        required_vars = {
            "REDIS_URL": os.environ.get("REDIS_URL"),
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "COHERE_API_KEY": os.environ.get("COHERE_API_KEY"),
        }

        missing_vars = {k: v for k, v in required_vars.items() if not v}

        if missing_vars:
            self.credentials_set = False
        else:
            self.credentials_set = True

        self.initialized = False
        self.RERANKERS = {}

        # Initialize non-API dependent variables
        self.chunk_size = int(os.environ.get("DEFAULT_CHUNK_SIZE", 500))
        self.chunking_technique = os.environ.get("DEFAULT_CHUNKING_TECHNIQUE", "Recursive Character")
        self.chain = None
        self.chat_history = None
        self.N = 0
        self.count = 0
        self.use_semantic_cache = str_to_bool(os.environ.get("DEFAULT_USE_SEMANTIC_CACHE"))
        self.use_rerankers = str_to_bool(os.environ.get("DEFAULT_USE_RERANKERS"))
        self.top_k = int(os.environ.get("DEFAULT_TOP_K", 3))
        self.distance_threshold = float(os.environ.get("DEFAULT_DISTANCE_THRESHOLD", 0.30))
        self.llm_temperature = float(os.environ.get("DEFAULT_LLM_TEMPERATURE", 0.7))
        self.use_chat_history = str_to_bool(os.environ.get("DEFAULT_USE_CHAT_HISTORY"))
        self.use_semantic_router = str_to_bool(os.environ.get("DEFAULT_USE_SEMANTIC_ROUTER"))
        self.use_ragas = str_to_bool(os.environ.get("DEFAULT_USE_RAGAS"))

        self.available_llms = {
            "openai": sorted(openai_models()),
        }

        if self.azure_openai_deployment is not None:
            self.available_llms["azure-openai"] = [self.azure_openai_deployment]

        self.llm_model_providers = list(self.available_llms.keys())
        self.selected_llm_provider = "openai"
        self.selected_llm = "gpt-3.5-turbo"

        self.available_embedding_models = {
            "openai": ["text-embedding-ada-002", "text-embedding-3-small"],
        }

        if self.azure_openai_deployment is not None:
            self.available_embedding_models["azure-openai"] = ["text-embedding-ada-002", "text-embedding-3-small"]

        self.embedding_model_providers = list(self.available_embedding_models.keys())
        self.selected_embedding_model_provider = "openai"
        self.selected_embedding_model = "text-embedding-ada-002"

        self.llm = None
        self.evalutor_llm = None
        self.cached_llm = None
        self.vector_store = None
        self.llmcache = None
        self.index_name = None

        if self.credentials_set:
            self.initialize_components()

    def initialize_components(self):
        if not self.credentials_set:
            raise ValueError("Credentials must be set before initializing components")

        self.pdf_manager = PDFManager(self.redis_url)

        # Initialize rerankers
        self.RERANKERS = {
            "HuggingFace": HFCrossEncoderReranker("BAAI/bge-reranker-base"),
            "Cohere": CohereReranker(
                limit=3, api_config={"api_key": self.cohere_api_key}
            ),
        }

        # Init semantic router
        self.semantic_router = SemanticRouter.from_yaml("demos/workbench/router.yaml", redis_url=self.redis_url, overwrite=True)

        # Init chat history if use_chat_history is True
        if self.use_chat_history:
            self.chat_history = RedisChatMessageHistory(
                session_id=self.session_id, redis_url=self.redis_url
            )
        else:
            self.chat_history = None

        # Init LLM
        self.update_llm()

        self.initialized = True


    def initialize_session(self):
        self.session_id = str(ULID())
        if self.use_chat_history:
            self.chat_history = RedisChatMessageHistory(
                session_id=self.session_id,
                redis_url=self.redis_url,
                index_name="idx:chat_history",  # Use a common index for all chat histories
            )
        else:
            self.chat_history = None

        return {"session_id": self.session_id, "chat_history": self.chat_history}

    def set_credentials(self, redis_url, openai_key, cohere_key):
        self.redis_url = redis_url or self.redis_url
        self.openai_api_key = openai_key or self.openai_api_key
        self.cohere_api_key = cohere_key or self.cohere_api_key

        # Update environment variables
        os.environ["REDIS_URL"] = self.redis_url
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        os.environ["COHERE_API_KEY"] = self.cohere_api_key

        self.credentials_set = all(
            [self.redis_url, self.openai_api_key, self.cohere_api_key]
        )

        if self.credentials_set:
            self.initialize_components()

        return "Credentials updated successfully. You can now use the demo."

    def get_llm(self):
        """Get the right LLM based on settings and config."""
        if self.selected_llm_provider == "azure-openai":
            try:
                model = AzureChatOpenAI(
                    azure_deployment=self.selected_llm,
                    api_version=self.azure_openai_api_version,
                    api_key=self.azure_openai_api_key,
                    azure_endpoint=self.azure_openai_endpoint,
                    temperature=self.llm_temperature,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                )
            except Exception as e:
                raise ValueError(
                    f"Error initializing Azure OpenAI model: {e} - must provide credentials for deployment"
                )
        else:
            model = ChatOpenAI(
                model=self.selected_llm,
                temperature=0,
            )

        return model

    def get_embedding_model(self):
        """Get the right embedding model based on settings and config"""
        if self.selected_embedding_model_provider == "azure-openai":
            return AzureOpenAIEmbeddings(
                model=self.selected_embedding_model,
                api_key=self.azure_openai_api_key,
                api_version=self.azure_openai_api_version,
                azure_endpoint=self.azure_openai_endpoint,
            )
        else:
            return OpenAIEmbeddings(api_key=self.openai_api_key)

    def get_reranker_choices(self):
        if self.initialized:
            return list(self.RERANKERS.keys())
        return ["HuggingFace", "Cohere"]  # Default choices before initialization

    def __call__(self, file: str, chunk_size: int, chunking_technique: str) -> Any:
        """Process a file upload directly - used by the UI."""
        self.chunk_size = chunk_size
        self.chunking_technique = chunking_technique

        # First store the PDF and get its index
        return self.process_pdf(file, chunk_size, chunking_technique)

    def build_chain(self, vector_store):
        retriever = vector_store.as_retriever(search_kwargs={"k": self.top_k})

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant. Use the following pieces of context to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.",
                ),
                ("system", "Context: {context}"),
                ("human", "{input}"),
                (
                    "system",
                    "Provide a helpful and accurate answer based on the given context and question:",
                ),
            ]
        )

        combine_docs_chain = create_stuff_documents_chain(self.cached_llm, prompt)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        return rag_chain

    def update_chat_history(self, use_chat_history: bool, session_state):
        self.use_chat_history = use_chat_history

        if session_state is None:
            session_state = self.initialize_session()

        if self.use_chat_history:
            if (
                "chat_history" not in session_state
                or session_state["chat_history"] is None
            ):
                session_state["chat_history"] = RedisChatMessageHistory(
                    session_id=session_state.get("session_id", str(ULID())),
                    redis_url=self.redis_url,
                    index_name="idx:chat_history",
                )

            try:
                messages_count = len(session_state["chat_history"].messages)
            except Exception as e:
                print(f"DEBUG: Error getting chat history length: {str(e)}")
        else:
            if "chat_history" in session_state and session_state["chat_history"]:
                try:
                    session_state["chat_history"].clear()
                except Exception as e:
                    print(f"DEBUG: Error clearing chat history: {str(e)}")
            session_state["chat_history"] = None

        return session_state

    def get_chat_history(self):
        if self.chat_history and self.use_chat_history:
            messages = self.chat_history.messages
            formatted_history = []
            for msg in messages:
                if msg.type == "human":
                    formatted_history.append(f"👤 **Human**: {msg.content}\n")
                elif msg.type == "ai":
                    formatted_history.append(f"🤖 **AI**: {msg.content}\n")
            return "\n".join(formatted_history)
        return "No chat history available."

    def update_semantic_router(self, use_semantic_router: bool):
        self.use_semantic_router = use_semantic_router

    def update_ragas(self, use_ragas: bool):
        self.use_ragas = use_ragas

    def update_llm(self):
        self.llm = self.get_llm()
        self.evalutor_llm = LangchainLLMWrapper(self.llm)

        if self.use_semantic_cache:
            self.cached_llm = CachedLLM(self.llm, self.llmcache)
        else:
            self.cached_llm = self.llm

        # update the chain with the new model
        # TODO: probably a better way to manage the lifecycle than to check the null because that could lead to odd error states
        if self.vector_store:
            self.chain = self.build_chain(self.vector_store)

    def update_model(self, new_model: str, new_model_provider: str):
        self.selected_llm = new_model
        self.selected_llm_provider = new_model_provider
        self.update_llm()

    def update_temperature(self, new_temperature: float):
        self.llm_temperature = new_temperature
        self.update_llm()

    def update_top_k(self, new_top_k: int):
        self.top_k = new_top_k

    def make_semantic_cache(self) -> SemanticCache:
        semantic_cache_index_name = f"llmcache:{self.index_name}"
        return SemanticCache(
            name=semantic_cache_index_name,
            redis_url=self.redis_url,
            distance_threshold=self.distance_threshold,
        )
    
    def clear_semantic_cache(self):
        # Always make a new SemanticCache in case use_semantic_cache is False
        semantic_cache = self.make_semantic_cache()
        semantic_cache.clear()


    def update_semantic_cache(self, use_semantic_cache: bool):
        self.use_semantic_cache = use_semantic_cache
        if self.use_semantic_cache and self.index_name:
            self.llmcache = self.make_semantic_cache()
        else:
            self.llmcache = None

        self.update_llm()

    def update_distance_threshold(self, new_threshold: float):
        self.distance_threshold = new_threshold
        if self.index_name:
            self.llmcache = self.make_semantic_cache()
            self.update_llm()

    def get_last_cache_status(self) -> bool:
        if isinstance(self.cached_llm, CachedLLM):
            return self.cached_llm.get_last_cache_status()
        return False

    def rerank_results(self, query, results):
        if not self.use_reranker:
            return results, None, None

        reranker = self.RERANKERS[self.reranker_type]
        original_results = [r.page_content for r in results]

        reranked_results, scores = reranker.rank(query=query, docs=original_results)

        # Reconstruct the results with reranked order, using fuzzy matching
        reranked_docs = []
        for reranked in reranked_results:
            reranked_content = (
                reranked["content"] if isinstance(reranked, dict) else reranked
            )
            best_match = max(
                results, key=lambda r: self.similarity(r.page_content, reranked_content)
            )
            reranked_docs.append(best_match)

        rerank_info = {
            "original_order": original_results,
            "reranked_order": [
                r["content"] if isinstance(r, dict) else r for r in reranked_results
            ],
            "original_scores": [1.0]
            * len(results),  # Assuming original scores are not available
            "reranked_scores": scores,
        }

        return reranked_docs, rerank_info, original_results

    def similarity(self, s1, s2):
        # Simple similarity measure based on common words
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        return len(words1.intersection(words2)) / len(words1.union(words2))

    def rerankers(self):
        return self.RERANKERS

    def evaluate_response(self, query, result):
        ds = Dataset.from_dict(
            {
                "question": [query],
                "answer": [result["answer"]],
                "contexts": [[c.page_content for c in result["context"]]],
            }
        )

        try:
            eval_results = evaluate(
                dataset=ds,
                metrics=[faithfulness, answer_relevancy],
                llm=self.evalutor_llm
            )

            return eval_results
        except Exception as e:
            print(f"Error during RAGAS evaluation: {e}")
            return {}

    def update_embedding_model_provider(self, new_provider: str):
        self.selected_embedding_model_provider = new_provider

    def process_pdf(
        self, file, chunk_size: int, chunking_technique: str, selected_embedding_model: str
    ) -> Any:
        """Process a new PDF file upload."""
        try:
            # First process the file to get documents
            documents, _ = process_file(file, chunk_size, chunking_technique)

            # Store the PDF and metadata first
            self.current_pdf_index = self.pdf_manager.add_pdf(
                file=file,
                chunk_size=chunk_size,
                chunking_technique=chunking_technique,
                total_chunks=len(documents),
            )

            # Set the index name from the PDF manager
            self.index_name = self.current_pdf_index
            self.selected_embedding_model = selected_embedding_model

            # Create the vector store using the same index
            embeddings = self.get_embedding_model()
            self.vector_store = RedisVectorStore.from_documents(
                documents,
                embeddings,
                redis_url=self.redis_url,
                index_name=self.index_name,
            )

            self.update_semantic_cache(self.use_semantic_cache)
            self.chain = self.build_chain(self.vector_store)
            return self.chain

        except Exception as e:
            print(f"Error during process_pdf: {e}")

    def load_pdf(self, index_name: str) -> bool:
        """Load a previously processed PDF."""
        try:
            # Get the metadata
            metadata = self.pdf_manager.get_pdf_metadata(index_name)
            if not metadata:
                return False

            # Set the current state
            self.current_pdf_index = index_name
            self.index_name = index_name
            self.chunk_size = metadata.chunk_size
            self.chunking_technique = metadata.chunking_technique

            # Set up vector store with embeddings as first argument
            # TODO: the embedding model probably needs to get store in the index for the loading option
            embeddings = self.get_embedding_model()
            self.vector_store = RedisVectorStore(
                embeddings,
                redis_url=self.redis_url,
                index_name=self.current_pdf_index,
            )

            # Update semantic cache if enabled
            self.update_semantic_cache(self.use_semantic_cache)

            # Build the chain
            self.chain = self.build_chain(self.vector_store)

            return True

        except Exception as e:
            print(f"ERROR: Failed to load PDF {index_name}: {str(e)}")
            return False

    def search_pdfs(self, query: str = "") -> List[PDFMetadata]:
        """Search available PDFs."""
        return self.pdf_manager.search_pdfs(query)

    def get_pdf_file(self, index_name: str) -> Optional[str]:
        """Get the file path for a stored PDF."""
        return self.pdf_manager.get_pdf_file(index_name)


def generate_feedback(evaluation_scores):
    if not evaluation_scores:
        return "RAGAS evaluation failed."

    feedback = ["RAGAS Metrics:"]
    for metric, score in evaluation_scores.items():
        feedback.append(f"  - {metric}: {score:.4f}")
    return "\n".join(feedback)
