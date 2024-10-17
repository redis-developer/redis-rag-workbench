import time

import gradio as gr
from gradio_modal import Modal
from langchain_community.callbacks import get_openai_callback

from demos.chat_with_pdf.chat_app import ChatApp, generate_feedback
from shared_components.pdf_utils import render_file, render_first_page
from shared_components.theme_management import load_theme

# app to be used in the gradio app
app = ChatApp()
redis_theme, redis_styles = load_theme("redis")


# functions for use in main
def path():
    return "/chat_with_pdf"


def app_title():
    return "Chat with one PDF"


# gradio functions define what happens with certain UI elements
def add_text(history, text: str):
    if not text:
        raise gr.Error("enter text")
    history = history + [(text, "")]
    return history


def reset_app():
    app.chat_history = []
    app.N = 0
    return [], None, "", gr.update(visible=False)


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

    is_cache_hit = False

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


# gradio FE
with gr.Blocks(theme=redis_theme, css=redis_styles) as demo:
    session_state = gr.State()

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
