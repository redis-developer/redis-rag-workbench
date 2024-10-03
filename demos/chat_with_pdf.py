import os
import os.path
import time
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from gradio_modal import Modal
from langchain.chains import RetrievalQA
from langchain_community.callbacks import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_redis import RedisChatMessageHistory, RedisVectorStore

# from ragas.integrations.langchain import EvaluatorChain TODO: decide if we can delete the other code
from ragas.metrics import answer_relevancy, faithfulness
from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.rerank import CohereReranker, HFCrossEncoderReranker
from ulid import ULID
from redis.exceptions import ResponseError

from shared_components.cached_llm import CachedLLM
from shared_components.llm_utils import openai_models
from shared_components.pdf_utils import process_file, render_file, render_first_page
from shared_components.theme_management import load_theme

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from datasets import Dataset

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)

from ragas import evaluate

# this isn't supported yet with new version https://stackoverflow.com/questions/77982768/no-module-named-ragas-langchain
# from ragas.langchain import RagasEvaluatorChain

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


class MyApp:
    def __init__(self) -> None:
        self.session_id = None
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

        # Initialize rerankers
        self.RERANKERS = {
            "HuggingFace": HFCrossEncoderReranker("BAAI/bge-reranker-base"),
            "Cohere": CohereReranker(
                limit=3, api_config={"api_key": self.cohere_api_key}
            ),
        }

        self.openai_client = OpenAI(api_key=self.openai_api_key)

        # # Initialize SemanticCache
        # self.llmcache = SemanticCache(
        #     name=f"llmcache:{self.session_id}",
        #     redis_url=self.redis_url,
        #     distance_threshold=self.distance_threshold,
        # )

        # Initialize chat history if use_chat_history is True
        if self.use_chat_history:
            self.chat_history = RedisChatMessageHistory(
                session_id=self.session_id, redis_url=self.redis_url
            )
        else:
            self.chat_history = None

        self.initialized = True

        # self.ensure_index_created()
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

    def update_llm(self):
        print("DEBUG: Updating LLM")
        self.llm = ChatOpenAI(
            model=self.selected_model,
            temperature=self.llm_temperature,
            api_key=self.openai_api_key,
        )

        if self.use_semantic_cache and self.llmcache:
            print("DEBUG: Using semantic cache")
            self.cached_llm = CachedLLM(self.llm, self.llmcache)
        else:
            print("DEBUG: Not using semantic cache")
            self.cached_llm = self.llm

        print(f"DEBUG: Updated LLM type: {type(self.cached_llm)}")

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
        self.chunk_size = chunk_size
        self.chunking_technique = chunking_technique
        documents, file_name = process_file(
            file, self.chunk_size, self.chunking_technique
        )
        self.index_name = "".join(
            c if c.isalnum() else "_" for c in file_name.replace(" ", "_")
        ).rstrip("_")

        print(f"DEBUG: Creating vector store with index name: {self.index_name}")
        embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        self.vector_store = RedisVectorStore.from_documents(
            documents,
            embeddings,
            redis_url=self.redis_url,
            index_name=self.index_name,
        )

        self.update_semantic_cache(self.use_semantic_cache)

        self.chain = self.build_chain(self.vector_store)
        return self.chain

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

        print("DEBUG: RAG chain created successfully")
        return rag_chain

    def update_chat_history(self, use_chat_history: bool, session_state):
        print(f"DEBUG: Updating chat history. use_chat_history: {use_chat_history}")
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
                print(
                    f"DEBUG: Created new RedisChatMessageHistory with session_id: {session_state['session_id']}"
                )

            print("DEBUG: Using existing RedisChatMessageHistory")
            try:
                messages_count = len(session_state["chat_history"].messages)
                print(f"DEBUG: Current chat history length: {messages_count}")
            except Exception as e:
                print(f"DEBUG: Error getting chat history length: {str(e)}")
        else:
            if "chat_history" in session_state and session_state["chat_history"]:
                try:
                    messages_count = len(session_state["chat_history"].messages)
                    print(
                        f"DEBUG: Clearing chat history. Current length: {messages_count}"
                    )
                    session_state["chat_history"].clear()
                    print("DEBUG: Cleared existing chat history")
                except Exception as e:
                    print(f"DEBUG: Error clearing chat history: {str(e)}")
            session_state["chat_history"] = None

        print(f"DEBUG: Chat history setting updated to {self.use_chat_history}")
        return session_state

    def get_chat_history(self):
        if self.chat_history and self.use_chat_history:
            messages = self.chat_history.messages
            print(f"DEBUG: Retrieved {len(messages)} messages from chat history")
            formatted_history = []
            for msg in messages:
                if msg.type == "human":
                    formatted_history.append(f"👤 **Human**: {msg.content}\n")
                elif msg.type == "ai":
                    formatted_history.append(f"🤖 **AI**: {msg.content}\n")
            return "\n".join(formatted_history)
        return "No chat history available."

    def update_llm(self):
        print("DEBUG: Updating LLM")
        if self.llm is None:
            print("DEBUG: self.llm is None, initializing new LLM")
            self.llm = ChatOpenAI(
                model=self.selected_model,
                temperature=self.llm_temperature,
                api_key=self.openai_api_key,
            )

        if self.use_semantic_cache:
            print("DEBUG: Using semantic cache")
            self.cached_llm = CachedLLM(self.llm, self.llmcache)
        else:
            print("DEBUG: Not using semantic cache")
            self.cached_llm = self.llm

        print(f"DEBUG: Updated LLM type: {type(self.cached_llm)}")

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
        if self.vector_store:
            # Update the search_kwargs for the existing retriever
            self.vector_store.search_kwargs["k"] = self.top_k

    def update_semantic_cache(self, use_semantic_cache: bool):
        print(
            f"DEBUG: Updating semantic cache. use_semantic_cache: {use_semantic_cache}"
        )
        self.use_semantic_cache = use_semantic_cache
        if self.use_semantic_cache and self.index_name:
            semantic_cache_index_name = f"llmcache:{self.index_name}"
            self.llmcache = SemanticCache(
                name=semantic_cache_index_name,
                redis_url=self.redis_url,
                distance_threshold=self.distance_threshold,
            )
            print(
                f"DEBUG: Created SemanticCache with index: {semantic_cache_index_name}"
            )

            # Ensure the index is created
            try:
                self.llmcache._index.info()
            except ResponseError:
                print(
                    f"DEBUG: Creating new index for SemanticCache: {semantic_cache_index_name}"
                )
                self.llmcache._index.create()

            self.update_llm()
            if self.vector_store:
                self.chain = self.build_chain(self.vector_store)
        else:
            self.llmcache = None
            print("DEBUG: SemanticCache disabled")

        print(f"DEBUG: Updated semantic cache setting to {use_semantic_cache}")

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
    use_chat_history,
    session_state,
):
    if not session_state:
        app.session_state = app.initialize_session()
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

    chain = app.chain
    start_time = time.time()

    print(f"DEBUG: Invoking chain with query: {query}")
    print(f"DEBUG: use_chat_history: {app.use_chat_history}")

    with get_openai_callback() as cb:
        result = chain.invoke({"input": query})
        end_time = time.time()

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

    if app.use_chat_history and session_state["chat_history"] is not None:
        session_state["chat_history"].add_user_message(query)
        session_state["chat_history"].add_ai_message(result["answer"])
        try:
            print(
                f"DEBUG: Added to chat history. Current length: {len(session_state['chat_history'].messages)}"
            )
            print(
                f"DEBUG: Last message in history: {session_state['chat_history'].messages[-1].content[:50]}..."
            )
        except Exception as e:
            print(f"DEBUG: Error accessing chat history: {str(e)}")
    else:
        print("DEBUG: Chat history not updated (disabled or None)")

    # Prepare the output
    if is_cache_hit:
        output = f"⏱️ | Cache: {elapsed_time:.2f} SEC | COST $0.00"
    else:
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
        output = f"⏱️ | LLM: {elapsed_time:.2f} SEC | {tokens_per_sec:.2f} TOKENS/SEC | {num_tokens} TOKENS | COST ${total_cost:.4f}"

    # Yield the response and output
    for char in result["answer"]:
        history[-1][-1] += char
        yield history, "", output, session_state

    # Perform RAGAS evaluation after yielding the response
    feedback = perform_ragas_evaluation(query, result)

    # Prepare the final output with RAGAS evaluation
    final_output = f"{output}\n\n{feedback}"

    # Yield one last time to update with RAGAS evaluation results
    yield history, "", final_output, session_state


def generate_feedback(evaluation_scores):
    if not evaluation_scores:
        return "RAGAS evaluation failed."

    feedback = ["RAGAS Metrics:"]
    for metric, score in evaluation_scores.items():
        feedback.append(f"  - {metric}: {score:.4f}")
    return "\n".join(feedback)


def render_first(file, chunk_size, chunking_technique, session_state):
    if not session_state:
        session_state = app.initialize_session()
    image = render_first_page(file)
    # Create the chain when the PDF is uploaded, using the specified chunk size and technique
    app.chain = app(file, chunk_size, chunking_technique)
    print(
        f"DEBUG: Chain created in render_first with chunk size {chunk_size} and {chunking_technique}"
    )
    return image, [], session_state


# Connect the show_history_btn to the display_chat_history function and show the modal
def show_history(session_state):
    print(f"DEBUG: show_history called. use_chat_history: {app.use_chat_history}")
    if app.use_chat_history and session_state["chat_history"] is not None:
        try:
            messages = session_state["chat_history"].messages
            print(f"DEBUG: Retrieved {len(messages)} messages from chat history")
            formatted_history = []
            for msg in messages:
                if msg.type == "human":
                    formatted_history.append(f"👤 **Human**: {msg.content}\n")
                elif msg.type == "ai":
                    formatted_history.append(f"🤖 **AI**: {msg.content}\n")
            history = "\n".join(formatted_history)
        except Exception as e:
            print(f"DEBUG: Error retrieving chat history: {str(e)}")
            history = "Error retrieving chat history."
    else:
        history = "No chat history available."

    print(f"DEBUG: Formatted chat history: {history[:100]}...")
    return history, gr.update(visible=True)


def reset_app():
    app.chat_history = []
    app.N = 0
    return [], None, "", gr.update(visible=False)


app = MyApp()
redis_theme, redis_styles = load_theme("redis")

with gr.Blocks(theme=redis_theme, css=redis_styles + _LOCAL_CSS) as demo:
    session_state = gr.State()

    gr.HTML(
        "<button class='primary' onclick=\"window.location.href='/demos'\">Back to Demos</button>"
    )

    # Add Modal for credentials input
    with Modal(visible=False) as credentials_modal:
        gr.Markdown("## Enter Missing Credentials")
        redis_url_input = gr.Textbox(
            label="REDIS_URL", type="password", value=app.redis_url or ""
        )
        openai_key_input = gr.Textbox(
            label="OPENAI_API_KEY", type="password", value=app.openai_api_key or ""
        )
        cohere_key_input = gr.Textbox(
            label="COHERE_API_KEY", type="password", value=app.cohere_api_key or ""
        )
        credentials_status = gr.Markdown("Please enter the missing credentials.")
        submit_credentials_btn = gr.Button("Submit Credentials")

    with gr.Row():
        # Left Half
        with gr.Column(scale=6):
            chatbot = gr.Chatbot(value=[], elem_id="chatbot")
            feedback_markdown = gr.Markdown(
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
                use_semantic_cache = gr.Checkbox(
                    label="Use Semantic Cache", value=app.use_semantic_cache
                )
                distance_threshold = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=app.distance_threshold,
                    step=0.01,
                    label="Distance Threshold",
                )

            with gr.Row():
                use_reranker = gr.Checkbox(
                    label="Use Reranker", value=app.use_rerankers
                )
                reranker_type = gr.Dropdown(
                    choices=list(app.rerankers().keys()),
                    label="Reranker Type",
                    value="HuggingFace",
                    interactive=True,
                )

            with gr.Row():
                use_chat_history = gr.Checkbox(
                    label="Use Chat History", value=app.use_chat_history
                )
                show_history_btn = gr.Button("Show Chat History")

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

    # Add Modal for chat history here, outside of any column or row
    with Modal(visible=False) as history_modal:
        # gr.Markdown("Chat History")
        history_display = gr.Markdown("No chat history available.")

    btn.upload(
        fn=render_first,
        inputs=[btn, chunk_size, chunking_technique, session_state],
        outputs=[show_img, chatbot, session_state],
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
            use_chat_history,
            session_state,
        ],
        outputs=[chatbot, txt, feedback_markdown, session_state],
    ).success(
        fn=render_file, inputs=[btn], outputs=[show_img]
    )

    reset_btn.click(
        fn=reset_app,
        inputs=None,
        outputs=[chatbot, show_img, txt, feedback_markdown],
    )

    use_chat_history.change(
        fn=app.update_chat_history,
        inputs=[use_chat_history, session_state],
        outputs=[session_state],
    )

    show_history_btn.click(
        fn=show_history,
        inputs=[session_state],
        outputs=[history_display, history_modal],
    )

    # Event handlers
    def check_credentials():
        if not app.credentials_set:
            return gr.update(visible=True)
        return gr.update(visible=False)

    demo.load(check_credentials, outputs=credentials_modal)

    def update_components_state():
        return [
            gr.update(interactive=app.credentials_set and app.initialized)
            for _ in range(4)
        ]

    # handle toggle of the semantic cache usage
    use_semantic_cache.change(
        fn=app.update_semantic_cache, inputs=[use_semantic_cache], outputs=[]
    )

    # Use success event to update components state after submitting credentials
    submit_credentials_btn.click(
        fn=app.set_credentials,
        inputs=[redis_url_input, openai_key_input, cohere_key_input],
        outputs=credentials_status,
    ).then(
        fn=check_credentials,
        outputs=credentials_modal,
    ).then(
        fn=lambda: gr.update(choices=app.get_reranker_choices()),
        outputs=reranker_type,
    ).then(
        fn=update_components_state,
        outputs=[txt, submit_btn, btn, reset_btn],
    )

    # Use success event to update components state after submitting credentials
    submit_credentials_btn.click(
        fn=update_components_state,
        outputs=[txt, submit_btn, btn, reset_btn],
    )
