import os
import os.path
from typing import Any, List, Optional

from datasets import Dataset
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_redis import RedisChatMessageHistory, RedisVectorStore
from ragas import evaluate

from ragas.metrics import answer_relevancy, faithfulness
from redis.exceptions import ResponseError
from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.rerank import CohereReranker, HFCrossEncoderReranker
from ulid import ULID

from shared_components.cached_llm import CachedLLM
from shared_components.llm_utils import openai_models
from shared_components.pdf_manager import PDFManager, PDFMetadata
from shared_components.pdf_utils import process_file

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
        self.chunk_size = 500
        self.chunking_technique = "Recursive Character"
        self.chain = None
        self.chat_history = None
        self.N = 0
        self.count = 0
        self.use_semantic_cache = False
        self.use_rerankers = False
        self.top_k = 1
        self.distance_threshold = 0.30
        self.selected_model = "gpt-3.5-turbo"
        self.llm_temperature = 0.7
        self.use_chat_history = False

        self.available_models = sorted(openai_models())
        self.llm = None
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

        self.openai_client = OpenAI(api_key=self.openai_api_key)

        # Initialize chat history if use_chat_history is True
        if self.use_chat_history:
            self.chat_history = RedisChatMessageHistory(
                session_id=self.session_id, redis_url=self.redis_url
            )
        else:
            self.chat_history = None

        self.initialized = True

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

    def get_reranker_choices(self):
        if self.initialized:
            return list(self.RERANKERS.keys())
        return ["HuggingFace", "Cohere"]  # Default choices before initialization

    def ensure_index_created(self):
        try:
            self.llmcache._index.info()
        except:
            self.llmcache._index.create()

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

    def update_llm(self):
        if self.llm is None:
            self.llm = ChatOpenAI(
                model=self.selected_model,
                temperature=self.llm_temperature,
                api_key=self.openai_api_key,
            )

        if self.use_semantic_cache:
            self.cached_llm = CachedLLM(self.llm, self.llmcache)
        else:
            self.cached_llm = self.llm

    def update_model(self, new_model: str):
        self.selected_model = new_model
        self.llm = ChatOpenAI(
            model=self.selected_model,
            temperature=self.llm_temperature,
            api_key=self.openai_api_key,
        )
        self.update_llm()

    def update_temperature(self, new_temperature: float):
        self.llm_temperature = new_temperature
        self.llm = ChatOpenAI(
            model=self.selected_model,
            temperature=self.llm_temperature,
            api_key=self.openai_api_key,
        )
        self.update_llm()

    def update_top_k(self, new_top_k: int):
        self.top_k = new_top_k

    def update_semantic_cache(self, use_semantic_cache: bool):
        self.use_semantic_cache = use_semantic_cache
        if self.use_semantic_cache and self.index_name:
            semantic_cache_index_name = f"llmcache:{self.index_name}"
            self.llmcache = SemanticCache(
                name=semantic_cache_index_name,
                redis_url=self.redis_url,
                distance_threshold=self.distance_threshold,
            )

            # Ensure the index is created
            try:
                self.llmcache._index.info()
            except ResponseError:
                self.llmcache._index.create()

            self.update_llm()
            if self.vector_store:
                self.chain = self.build_chain(self.vector_store)
        else:
            self.llmcache = None

    def update_distance_threshold(self, new_threshold: float):
        self.distance_threshold = new_threshold
        if self.index_name:
            self.llmcache = SemanticCache(
                name=f"llmcache:{self.index_name}",
                redis_url=self.redis_url,
                distance_threshold=self.distance_threshold,
            )
            self.ensure_index_created()
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
            eval_results = evaluate(ds, [faithfulness, answer_relevancy])

            return eval_results
        except Exception as e:
            print(f"Error during RAGAS evaluation: {e}")
            return {}

    def process_pdf(self, file, chunk_size: int, chunking_technique: str) -> Any:
        """Process a new PDF file upload."""
        try:
            # First process the file to get documents
            documents, _ = process_file(file, chunk_size, chunking_technique)

            # Store the PDF and metadata first
            self.current_pdf_index = self.pdf_manager.add_pdf(
                file=file,
                chunk_size=chunk_size,
                chunking_technique=chunking_technique,
                total_chunks=len(documents)
            )

            # Set the index name from the PDF manager
            self.index_name = self.current_pdf_index

            # Create the vector store using the same index
            embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
            self.vector_store = RedisVectorStore.from_documents(
                documents,
                embeddings,
                redis_url=self.redis_url,
                index_name=self.index_name
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
            self.chunk_size = metadata.chunk_size
            self.chunking_technique = metadata.chunking_technique

            # Set up vector store with embeddings as first argument
            embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
            self.vector_store = RedisVectorStore(
                embeddings,  # First positional argument
                redis_url=self.redis_url,
                index_name=self.current_pdf_index
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
