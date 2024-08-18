import os
import os.path
import time
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_redis import RedisVectorStore
from ragas.integrations.langchain import EvaluatorChain
from ragas.metrics import answer_relevancy, faithfulness
from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.rerank import CohereReranker, HFCrossEncoderReranker

from shared_components.cached_llm import CachedLLM
from shared_components.pdf_utils import (process_file, render_file,
                                         render_first_page)
from shared_components.theme_management import load_theme
from shared_components.llm_utils import openai_models

load_dotenv()

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
    def __init__(self) -> None:
        self.redis_url = os.environ.get("REDIS_URL")
        self.openai_api_key: str = os.environ.get("OPENAI_API_KEY")
        self.cohere_api_key: str = os.environ.get("COHERE_API_KEY")

        # Initialize rerankers
        hf_reranker = HFCrossEncoderReranker("BAAI/bge-reranker-base")
        cohere_reranker = CohereReranker(
            limit=3, api_config={"api_key": self.cohere_api_key}
        )

        self.RERANKERS = {"HuggingFace": hf_reranker, "Cohere": cohere_reranker}

        # chunking settings
        self.chunk_size = 500
        self.chunking_technique = "Recursive Character"

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

        # Default Top K
        self.top_k = 1
        # Default SemanticCache distance threshold
        self.distance_threshold = 0.20

        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.available_models = sorted(openai_models())
        self.selected_model = "gpt-3.5-turbo"  # Default model

        # LLM settings
        self.llm_temperature = 0.7

        # Initialize RAGAS evaluator chains
        self.faithfulness_chain = EvaluatorChain(metric=faithfulness)
        self.answer_rel_chain = EvaluatorChain(metric=answer_relevancy)

        # Initialize SemanticCache
        self.llmcache = SemanticCache(
            index_name="llmcache",
            redis_url=self.redis_url,
            distance_threshold=self.distance_threshold,
        )
        self.ensure_index_created()

    def ensure_index_created(self):
        try:
            self.llmcache._index.info()
        except:
            self.llmcache._index.create()

    def __call__(self, file: str, chunk_size: int, chunking_technique: str) -> Any:
        self.chunk_size = chunk_size
        self.chunking_technique = chunking_technique
        self.chain = self.build_chain(file)
        return self.chain

    def build_chain(self, file: str):
        print(
            f"DEBUG: Starting build_chain for file: {file.name} with chunk size: {self.chunk_size} and {self.chunking_technique}"
        )
        documents, file_name = process_file(
            file, self.chunk_size, self.chunking_technique
        )
        index_name = "".join(
            c if c.isalnum() else "_" for c in file_name.replace(" ", "_")
        ).rstrip("_")

        print(f"DEBUG: Creating vector store with index name: {index_name}")
        # Load embeddings model
        embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        vector_store = RedisVectorStore.from_documents(
            documents,
            embeddings,
            redis_url=self.redis_url,
            index_name=index_name,
        )

        # Create the retriever with the initial top_k value
        self.vector_store = vector_store.as_retriever(search_kwargs={"k": self.top_k})

        # Configure the LLM
        self.llm = ChatOpenAI(
            model=self.selected_model,
            temperature=self.llm_temperature,
            api_key=self.openai_api_key,
        )
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
        chain = create_retrieval_chain(self.vector_store, self.document_chain)

        # Create the retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            self.cached_llm,  # Use cached_llm instead of llm
            retriever=self.vector_store,
            return_source_documents=True,
        )

        return self.qa_chain

    def update_llm(self):
        if self.use_semantic_cache:
            self.cached_llm = CachedLLM(self.llm, self.llmcache)
        else:
            self.cached_llm = self.llm

    def update_chain(self):
        if self.vector_store:
            self.qa_chain = RetrievalQA.from_chain_type(
                self.cached_llm,
                retriever=self.vector_store,
                return_source_documents=True,
            )

    def update_model(self, new_model: str):
        self.selected_model = new_model
        self.llm = ChatOpenAI(
            model=self.selected_model,
            temperature=self.llm_temperature,
            api_key=self.openai_api_key,
        )
        self.update_llm()
        self.update_chain()

    def update_temperature(self, new_temperature: float):
        self.llm_temperature = new_temperature
        self.llm = ChatOpenAI(
            model=self.selected_model,
            temperature=self.llm_temperature,
            api_key=self.openai_api_key,
        )
        self.update_llm()
        self.update_chain()

    def update_top_k(self, new_top_k: int):
        self.top_k = new_top_k
        if self.vector_store:
            # Update the search_kwargs for the existing retriever
            self.vector_store.search_kwargs["k"] = self.top_k
        self.update_chain()

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

    def update_distance_threshold(self, new_threshold: float):
        self.distance_threshold = new_threshold
        self.llmcache = SemanticCache(
            index_name="llmcache",
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


def perform_ragas_evaluation(query, result):
    evaluation_scores = app.evaluate_response(query, result)
    feedback = generate_feedback(evaluation_scores)
    return feedback


def get_response(
    history,
    query,
    file,
    use_semantic_cache,
    use_reranker,
    reranker_type,
    distance_threshold,
    top_k,
    llm_model,
    llm_temperature,
):
    if not file:
        raise gr.Error(message="Upload a PDF")

    # Update parameters if changed
    if app.top_k != top_k:
        app.update_top_k(top_k)
    if app.distance_threshold != distance_threshold:
        app.update_distance_threshold(distance_threshold)
    if app.selected_model != llm_model:
        app.update_model(llm_model)
    if app.llm_temperature != llm_temperature:
        app.update_temperature(llm_temperature)

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

    # Yield the response first
    for char in answer:
        history[-1][-1] += char
        if is_cache_hit:
            yield history, "", f"⏱️ | Cache: {elapsed_time:.2f} SEC | COST $0.00 \n\n{rerank_feedback}\n\nEvaluating..."
        else:
            tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
            yield history, "", f"⏱️ | LLM: {elapsed_time:.2f} SEC | {tokens_per_sec:.2f} TOKENS/SEC | {num_tokens} TOKENS | COST ${total_cost:.4f}\n\n{rerank_feedback}\n\nEvaluating..."

    # Perform RAGAS evaluation after yielding the response
    feedback = perform_ragas_evaluation(query, result)

    # Yield the final result with RAGAS evaluation
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


def render_first(file, chunk_size, chunking_technique):
    image = render_first_page(file)

    # Create the chain when the PDF is uploaded, using the specified chunk size and technique
    app.chain = app(file, chunk_size, chunking_technique)
    print(
        f"DEBUG: Chain created in render_first with chunk size {chunk_size} and {chunking_technique}"
    )

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
        # Left Half
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(value=[], elem_id="chatbot")
            elapsed_time_markdown = gr.Markdown(
                value="", label="Elapsed Time", visible=True
            )

            with gr.Row(elem_id="input-row"):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter text and press enter",
                    elem_id="txt",
                    scale=5,
                )
                submit_btn = gr.Button("🔍 Submit", elem_id="submit-btn", scale=1)

            with gr.Row():
                with gr.Row():
                    llm_model = gr.Dropdown(
                        choices=app.available_models,
                        value=app.selected_model,
                        label="LLM Model",
                        interactive=True,
                    )
                    llm_temperature = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=app.llm_temperature,
                        step=0.1,
                        label="LLM Temperature",
                    )
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=app.top_k,
                        step=1,
                        label="Top K",
                    )

            with gr.Row():
                use_semantic_cache = gr.Checkbox(label="Use Semantic Cache", value=True)
                distance_threshold = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=app.distance_threshold,
                    step=0.01,
                    label="Semantic Cache Distance Threshold",
                )

            with gr.Row():
                use_reranker = gr.Checkbox(label="Use Reranker", value=False)
                reranker_type = gr.Dropdown(
                    choices=list(app.rerankers().keys()),
                    label="Reranker Type",
                    value="HuggingFace",
                    interactive=True,
                )

        # Right Half
        with gr.Column(scale=6):
            show_img = gr.Image(label="Uploaded PDF")

            with gr.Row():
                chunking_technique = gr.Radio(
                    ["Recursive Character", "Semantic"],
                    label="Chunking Technique",
                    value=app.chunking_technique,
                )

            with gr.Row():
                chunk_size = gr.Slider(
                    minimum=100,
                    maximum=1000,
                    value=app.chunk_size,
                    step=50,
                    label="Chunk Size",
                    info="Size of document chunks for processing",
                )

            with gr.Row():
                btn = gr.UploadButton(
                    "📁 Upload PDF", file_types=[".pdf"], elem_id="upload-btn"
                )
                reset_btn = gr.Button("Reset", elem_id="reset-btn")

    btn.upload(
        fn=render_first,
        inputs=[btn, chunk_size, chunking_technique],
        outputs=[show_img, chatbot],
    )

    submit_btn.click(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False,
    ).success(
        fn=get_response,
        inputs=[
            chatbot,
            txt,
            btn,
            use_semantic_cache,
            use_reranker,
            reranker_type,
            distance_threshold,
            top_k,
            llm_model,
            llm_temperature,
        ],
        outputs=[chatbot, txt, elapsed_time_markdown],
    ).success(
        fn=render_file, inputs=[btn], outputs=[show_img]
    )

    reset_btn.click(
        fn=reset_app,
        inputs=None,
        outputs=[chatbot, show_img, txt, elapsed_time_markdown],
    )
