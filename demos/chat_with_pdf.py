import json
from typing import Any, Optional
import time
import yaml
import gradio as gr
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisVectorStore

from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import PyPDFLoader

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompt_values import StringPromptValue
from langchain_core.runnables import RunnableSequence

import fitz
from PIL import Image

import re
import os.path
from typing import Tuple

from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.rerank import HFCrossEncoderReranker, CohereReranker
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.config import RunnableConfig

from ragas.metrics import faithfulness, answer_relevancy
from ragas.integrations.langchain import EvaluatorChain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from shared_components.theme_management import load_theme
from shared_components.cached_llm import CachedLLM

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_ASSET_DIR = os.path.dirname(__file__) + "/assets"

_LOCAL_CSS = """
#txt {
    flex: 1;
}

#submit-btn {
    width: 50px;
}

#upload-btn, #reset-btn {
    width: 50%;
}
"""


def path():
    return "/chat_with_pdf"


def app_title():
    return "Chat with one PDF"


def add_text(history, text: str):
    if not text:
        raise gr.Error("enter text")
    history = history + [(text, "")]
    return history


class my_app:
    def __init__(self, config_path="./config.yaml") -> None:
        self.config = self.load_config(config_path)
        self.redis_url = self.config.get("redis_url")
        self.OPENAI_API_KEY: str = self.config.get("openai_api_key")
        os.environ["OPENAI_API_KEY"] = (
            self.OPENAI_API_KEY
        )  # shouldn't do this but RAGAS needs it!
        self.cohere_api_key: str = self.config.get("cohere_api_key")

        # Initialize rerankers
        hf_reranker = HFCrossEncoderReranker("BAAI/bge-reranker-base")
        cohere_reranker = CohereReranker(
            limit=3, api_config={"api_key": self.cohere_api_key}
        )

        self.RERANKERS = {"HuggingFace": hf_reranker, "Cohere": cohere_reranker}

        self.chain = None
        self.chat_history: list = []
        self.N: int = 0
        self.count: int = 0

        self.cached_llm = None
        self.last_is_cache_hit = False

        self.use_semantic_cache = True
        self.llm = None
        self.vector_store = None
        self.document_chain = None

        # Initialize RAGAS evaluator chains
        self.faithfulness_chain = EvaluatorChain(metric=faithfulness)
        self.answer_rel_chain = EvaluatorChain(metric=answer_relevancy)

        # Initialize SemanticCache
        self.llmcache = SemanticCache(
            index_name="llmcache", redis_url=self.redis_url, distance_threshold=0.1
        )
        self.ensure_index_created()

    def ensure_index_created(self):
        try:
            self.llmcache._index.info()
        except:
            self.llmcache._index.create()

    def __call__(self, file: str) -> Any:
        self.chain = self.build_chain(file)
        return self.chain

    def load_config(self, file_path):
        """
        Load configuration from a YAML file.

        Parameters:
            file_path (str): Path to the YAML configuration file.

        Returns:
            dict: Configuration as a dictionary.
        """
        with open(file_path, "r") as stream:
            try:
                config = yaml.safe_load(stream)
                return config
            except yaml.YAMLError as exc:
                print(f"Error loading configuration: {exc}")
                return None

    def process_file(self, file: str):
        loader = PyPDFLoader(file.name)
        documents = loader.load()
        pattern = r"/([^/]+)$"
        match = re.search(pattern, file.name)
        file_name = match.group(1)
        return documents, file_name

    def build_chain(self, file: str):
        print(f"DEBUG: Starting build_chain for file: {file.name}")
        documents, file_name = self.process_file(file)
        index_name = "".join(
            c if c.isalnum() else "_" for c in file_name.replace(" ", "_")
        ).rstrip("_")

        print(f"DEBUG: Creating vector store with index name: {index_name}")
        # Load embeddings model
        embeddings = OpenAIEmbeddings(api_key=self.OPENAI_API_KEY)
        self.vector_store = RedisVectorStore.from_documents(
            documents,
            embeddings,
            redis_url=self.redis_url,
            index_name=index_name,
        )

        # Configure the LLM
        self.llm = ChatOpenAI(temperature=0.7, api_key=self.OPENAI_API_KEY)
        self.update_llm()

        # Create a prompt template
        prompt = PromptTemplate.from_template(
            """You are a helpful assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}

    Question: {input}

    Helpful Answer:"""
        )
        # Create the document chain
        self.document_chain = create_stuff_documents_chain(self.cached_llm, prompt)

        # Create the retrieval chain
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 1})
        chain = create_retrieval_chain(retriever, self.document_chain)

        # Create the retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            self.cached_llm,  # Use cached_llm instead of llm
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True,
        )

        return self.qa_chain

    def update_llm(self):
        if self.use_semantic_cache:
            self.cached_llm = CachedLLM(self.llm, self.llmcache)
        else:
            self.cached_llm = self.llm

    def update_semantic_cache(self, use_semantic_cache: bool):
        self.use_semantic_cache = use_semantic_cache
        self.update_llm()
        if self.chain:
            # Update the LLM in the chain
            if isinstance(self.chain, RunnableSequence):
                for step in self.chain.steps:
                    if hasattr(step, "llm_chain"):
                        step.llm_chain.llm = self.cached_llm
                    elif hasattr(step, "llm"):
                        step.llm = self.cached_llm
            print(
                f"DEBUG: Updated LLM in chain, use_semantic_cache: {use_semantic_cache}"
            )

    def get_last_cache_status(self) -> bool:
        if isinstance(self.cached_llm, CachedLLM):
            return self.cached_llm.get_last_cache_status()
        return False

    def rerank_results(self, query, results):
        print(
            f"DEBUG: Re-ranking step - Active: {self.use_reranker}, Reranker: {self.reranker_type}"
        )

        if not self.use_reranker:
            print("DEBUG: Re-ranking skipped (not active)")
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
        eval_input = {
            "question": query,
            "answer": result["result"],
            "contexts": [doc.page_content for doc in result["source_documents"]],
        }

        try:
            faithfulness_score = self.faithfulness_chain(eval_input)["faithfulness"]
            answer_relevancy_score = self.answer_rel_chain(eval_input)[
                "answer_relevancy"
            ]

            return {
                "faithfulness": faithfulness_score,
                "answer_relevancy": answer_relevancy_score,
            }
        except Exception as e:
            print(f"Error during RAGAS evaluation: {e}")
            return {}


def get_response(history, query, file, use_semantic_cache, use_reranker, reranker_type):
    if not file:
        raise gr.Error(message="Upload a PDF")

    # Check if the semantic cache setting has changed
    if app.use_semantic_cache != use_semantic_cache:
        app.update_semantic_cache(use_semantic_cache)
        app.chain = app(file)  # Rebuild the chain

    app.use_reranker = use_reranker
    app.reranker_type = reranker_type

    chain = app.chain
    start_time = time.time()

    print(f"DEBUG: Invoking chain with query: {query}")
    with get_openai_callback() as cb:
        result = chain.invoke({"query": query})
        end_time = time.time()

        # Apply re-ranking if enabled
        rerank_info = None
        if app.use_reranker:
            print(f"DEBUG: Reranking with {reranker_type}")
            reranked_docs, rerank_info, original_results = app.rerank_results(
                query, result["source_documents"]
            )
            if reranked_docs:
                result["source_documents"] = reranked_docs
            else:
                print("DEBUG: Re-ranking produced no results")
        else:
            print("DEBUG: Re-ranking skipped")

        is_cache_hit = app.get_last_cache_status()

        if not is_cache_hit:
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            total_cost = cb.total_cost
            num_tokens = cb.total_tokens
        else:
            total_cost = 0
            num_tokens = 0

    print(f"Cache Hit: {is_cache_hit}")

    elapsed_time = end_time - start_time

    answer = result["result"]
    app.chat_history += [(query, answer)]

    # Run RAGAS evaluation
    evaluation_scores = app.evaluate_response(query, result)
    feedback = generate_feedback(evaluation_scores)

    # Prepare reranking feedback
    rerank_feedback = ""
    if rerank_info:
        original_order = rerank_info["original_order"]
        reranked_order = rerank_info["reranked_order"]
        reranked_scores = rerank_info["reranked_scores"]

        # Check if the order changed
        order_changed = original_order != reranked_order

        if order_changed:
            rerank_feedback = (
                f"ReRanking changed document order. Top score: {reranked_scores[0]:.4f}"
            )
        else:
            rerank_feedback = "ReRanking did not change document order."

    for char in answer:
        history[-1][-1] += char
        if is_cache_hit:
            yield history, "", f"⏱️ | Cache: {elapsed_time:.2f} SEC | COST $0.00 \n\n{rerank_feedback}\n\n{feedback}"
        else:
            tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
            yield history, "", f"⏱️ | LLM: {elapsed_time:.2f} SEC | {tokens_per_sec:.2f} TOKENS/SEC | {num_tokens} TOKENS | COST ${total_cost:.4f}\n\n{rerank_feedback}\n\n{feedback}"


def generate_feedback(evaluation_scores):
    if not evaluation_scores:
        return "RAGAS evaluation failed."

    feedback = ["RAGAS Metrics:"]
    for metric, score in evaluation_scores.items():
        feedback.append(f"  - {metric}: {score:.4f}")
    return "\n".join(feedback)


def render_file(file):
    doc = fitz.open(file.name)
    try:
        page = doc[app.N]
    except IndexError:
        print(f"Invalid page number: {app.N}, defaulting to page 0")
        page = doc[0]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return image


def render_first(file):
    doc = fitz.open(file.name)
    page = doc[0]
    # Render the page as a PNG image with a resolution of 300 DPI
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # Create the chain when the PDF is uploaded
    app.chain = app(file)
    print("DEBUG: Chain created in render_first")

    return image, []


def reset_app():
    app.chat_history = []
    app.N = 0
    return [], None, "", gr.update(visible=False)


app = my_app()
redis_theme, redis_styles = load_theme("redis")

with gr.Blocks(theme=redis_theme, css=redis_styles + _LOCAL_CSS) as demo:

    gr.HTML(
        "<button class='primary' onclick=\"window.location.href='/demos'\">Back to Demos</button>"
    )
    with gr.Row():
        with gr.Column(scale=6):
            with gr.Row():
                chatbot = gr.Chatbot(value=[], elem_id="chatbot")
            elapsed_time_markdown = gr.Markdown(
                value="", label="Elapsed Time", visible=True
            )
            with gr.Row():
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter",
                    elem_id="txt",
                    scale=1,
                )
                submit_btn = gr.Button("🔍 Submit", elem_id="submit-btn", scale=0)
            with gr.Row():
                use_semantic_cache = gr.Checkbox(label="Use Semantic Cache", value=True)
                use_reranker = gr.Checkbox(label="Use Reranker", value=False)
                reranker_type = gr.Dropdown(
                    choices=list(app.rerankers().keys()),
                    label="Reranker Type",
                    value="HuggingFace",
                    interactive=True,
                )
        with gr.Column(scale=6):
            show_img = gr.Image(label="Upload PDF")
            with gr.Row():
                btn = gr.UploadButton(
                    "📁 Upload PDF", file_types=[".pdf"], elem_id="upload-btn"
                )
                reset_btn = gr.Button("Reset", elem_id="reset-btn")

    btn.upload(
        fn=render_first,
        inputs=[btn],
        outputs=[show_img, chatbot],
    )

    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[
            chatbot,
        ],
        queue=False,
    ).success(
        fn=get_response,
        inputs=[chatbot, txt, btn, use_semantic_cache, use_reranker, reranker_type],
        outputs=[chatbot, txt, elapsed_time_markdown],
    ).success(
        fn=render_file, inputs=[btn], outputs=[show_img]
    )

    reset_btn.click(
        fn=reset_app,
        inputs=None,
        outputs=[chatbot, show_img, txt, elapsed_time_markdown],
    )
