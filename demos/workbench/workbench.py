import time
from datetime import datetime

import gradio as gr
from gradio_modal import Modal
from gradio_pdf import PDF
from langchain_community.callbacks import get_openai_callback

from demos.workbench.chat_app import ChatApp, generate_feedback
from shared_components.theme_management import load_theme

# app to be used in the gradio app
app = ChatApp()
redis_theme, redis_styles = load_theme("redis")


# functions for use in main
def path():
    return "/workbench"


def app_title():
    return "Chat with one PDF"


TAG_ESCAPE_CHARS = {
    ",",
    ".",
    "<",
    ">",
    "{",
    "}",
    "[",
    "]",
    '"',
    "'",
    ":",
    ";",
    "!",
    "@",
    "#",
    "$",
    "%",
    "^",
    "&",
    "*",
    "(",
    ")",
    "-",
    "+",
    "=",
    "~",
    "|",
    "/",
}


def escape_redis_search_query(query: str) -> str:
    return "".join(f"\\{char}" if char in TAG_ESCAPE_CHARS else char for char in query)


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
    if app.use_chat_history and session_state["chat_history"] is not None:
        try:
            messages = session_state["chat_history"].messages
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

    return history, gr.update(visible=True)


def render_first(file, chunk_size, chunking_technique, selected_embedding_model, session_state):
    """Handle initial PDF upload and rendering."""
    if not session_state:
        session_state = app.initialize_session()

    # First process and store the PDF properly
    app.process_pdf(file, chunk_size, chunking_technique, selected_embedding_model)

    # Create PDF viewer
    pdf_viewer = PDF(value=file.name, starting_page=1)

    return pdf_viewer, [], session_state


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
    selected_llm_provider,
    llm_temperature,
    use_chat_history,
    session_state,
):
    if not session_state:
        app.session_state = app.initialize_session()
    if not file and not app.chain:
        raise gr.Error(message="Please upload or select a PDF first")

    # Update parameters if changed
    # TODO: maybe change the naming convention of selected because it seems backwards to me
    if app.top_k != top_k:
        app.update_top_k(top_k)
    if app.distance_threshold != distance_threshold:
        app.update_distance_threshold(distance_threshold)
    if app.selected_llm != llm_model or app.selected_llm_provider != selected_llm_provider:
        app.update_model(
            llm_model, selected_llm_provider
        )  # was this passing the old model?
    if app.llm_temperature != llm_temperature:
        app.update_temperature(llm_temperature)

    chain = app.chain
    start_time = time.time()
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


def format_pdf_list(pdfs):
    """Format PDFs for display in the dataframe."""
    return [
        [
            pdf.filename,
            pdf.file_size,
            datetime.fromisoformat(pdf.upload_date).strftime("%Y-%m-%d %H:%M"),
        ]
        for pdf in pdfs
    ]


# UI event handlers
def update_pdf_list(search_query=""):
    """Update the PDF list based on search query."""
    pdfs = app.search_pdfs(f"*{search_query}*" if search_query else "*")
    return format_pdf_list(pdfs)


def handle_pdf_selection(evt: gr.SelectData, pdf_list):
    """Handle PDF selection from the list."""
    try:
        # Get the selected row using iloc for positional indexing
        selected_row = pdf_list.iloc[evt.index[0]]
        filename = selected_row.iloc[0]  # Use iloc for position-based access

        print(f"DEBUG: Selected PDF: {filename}")

        # Search for the PDF by filename
        filename = escape_redis_search_query(filename)
        pdfs = app.search_pdfs("@filename:{" + filename + "}")
        if not pdfs:
            print(f"DEBUG: PDF not found: {filename}")
            return None, [], "PDF not found", gr.update(visible=False)

        # Get the first matching PDF
        pdf = pdfs[0]
        print(f"DEBUG: Loading PDF with index: {pdf.index_name}")

        # Load the PDF into the app
        success = app.load_pdf(pdf.index_name)

        if not success:
            print("DEBUG: Failed to load PDF")
            return None, [], "Failed to load PDF", gr.update(visible=False)

        # Get the stored PDF file path
        pdf_path = app.get_pdf_file(pdf.index_name)
        if not pdf_path:
            print("DEBUG: PDF file not found")
            return None, [], "PDF file not found", gr.update(visible=False)

        # Render the first page for display
        print(f"DEBUG: Rendering first page of {pdf_path}")
        try:
            print(f"DEBUG: Loading PDF viewer for {pdf_path}")
            pdf_viewer = PDF(value=pdf_path, starting_page=1)
            return pdf_viewer, [], f"Loaded {filename}", gr.update(visible=False)
        except Exception as e:
            print(f"ERROR: Failed to render PDF: {str(e)}")
            return None, [], f"Error rendering PDF: {str(e)}", gr.update(visible=False)

    except Exception as e:
        print(f"ERROR: Failed to handle PDF selection: {str(e)}")
        return None, [], f"Error loading PDF: {str(e)}", gr.update(visible=False)


def handle_new_upload(file, chunk_size, chunking_technique, session_state):
    """Handle new PDF upload."""
    if not file:
        return None, [], None, gr.update(visible=True)

    # Process the file using the app's backend
    app.process_pdf(file, chunk_size, chunking_technique)

    # Create PDF viewer
    pdf_viewer = PDF(value=file.name, starting_page=1)

    return pdf_viewer, [], session_state, gr.update(visible=False)


def update_embedding_model(selected_embedding_model_provider):
    """Update the embedding model based on the selected provider."""
    app.update_embedding_model(selected_embedding_model_provider)
    return app.available_embedding_models[selected_embedding_model_provider]


HEADER = """
<div style="display: flex; justify-content: center;">
    <img src="../assets/redis-logo.svg" style="height: 2rem">
</div>
<div style="text-align: center">
    <h1>RAG Workbench</h1>
</div>
"""


def update_embedding_model_options(selected_embedding_model_provider, selected_embedding_model):
    # gradio has a weird thing where you have to include the second variable even if it's unused https://stackoverflow.com/questions/76693922/what-am-i-doing-wrong-with-gradio-dropdown-how-to-dynamically-modify-the-choice
    if app.selected_embedding_model_provider != selected_embedding_model_provider:
        app.update_embedding_model_provider(selected_embedding_model_provider)
        models = app.available_embedding_models[selected_embedding_model_provider]
        return gr.Dropdown(choices=models, value=models[0])


def update_llm_model_options(selected_llm_provider, llm_model):
    if app.selected_llm_provider != selected_llm_provider:
        app.update_model(llm_model, selected_llm_provider)
        models = app.available_llms[selected_llm_provider]
        return gr.Dropdown(choices=models, value=models[0])


HEADER = """
<div style="display: flex; justify-content: center;">
    <img src="../assets/redis-logo.svg" style="height: 2rem">
</div>
<div style="text-align: center">
    <h1>RAG Workbench</h1>
</div>
"""

HEADER = """
<div style="display: flex; justify-content: center;">
    <img src="../assets/redis-logo.svg" style="height: 2rem">
</div>
<div style="text-align: center">
    <h1>RAG Workbench</h1>
</div>
"""

# gradio FE
with gr.Blocks(theme=redis_theme, css=redis_styles, title="RAG Workbench") as demo:
    session_state = gr.State()

    # Add Modal for credentials input
    with Modal(visible=False) as credentials_modal:
        gr.Markdown("## Enter Missing Credentials")
        redis_url_input = gr.Textbox(
            label="REDIS_URL",
            type="password",
            value=app.redis_url or "redis://localhost:6379",
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
        gr.HTML(HEADER)

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
                    selected_llm_provider = gr.Dropdown(
                        choices=app.llm_model_providers,
                        value=app.selected_llm_provider,
                        label="LLM Model Provider",
                        # interactive=True,
                    )
                    llm_model = gr.Dropdown(
                        choices=app.available_llms[selected_llm_provider.value],
                        value=app.selected_llm,
                        label="LLM Model",
                        # interactive=True,
                    )

                    selected_llm_provider.change(
                        fn=update_llm_model_options,
                        inputs=[selected_llm_provider, llm_model],
                        outputs=[llm_model],
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
                        maximum=20,
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
            #show_pdf = PDF(label="Uploaded PDF", height=600)
            show_pdf = PDF(label="Uploaded PDF", height=600, elem_classes="pdf-parent")

            with gr.Row():
                selected_embedding_model_provider = gr.Dropdown(
                    choices=app.embedding_model_providers,
                    value=app.selected_embedding_model_provider,
                    label="Embedding Model Provider",
                    interactive=True,
                )

                selected_embedding_model = gr.Dropdown(
                    choices=app.available_embedding_models[selected_embedding_model_provider.value],
                    value=app.selected_embedding_model,
                    label="Embedding Model",
                    interactive=True,
                )

                selected_embedding_model_provider.change(
                    fn=update_embedding_model_options,
                    inputs=[selected_embedding_model_provider, selected_embedding_model],
                    outputs=[selected_embedding_model],
                )

            with gr.Row():
                chunking_technique = gr.Radio(
                    ["Recursive Character", "Semantic"],
                    label="Chunking Technique",
                    value=app.chunking_technique,
                )

            with gr.Row():
                chunk_size = gr.Slider(
                    minimum=100,
                    maximum=2500,
                    value=app.chunk_size,
                    step=50,
                    label="Chunk Size",
                    info="Size of document chunks for processing",
                )

            with gr.Row():
                select_pdf_btn = gr.Button("📄 Select PDF", elem_id="select-pdf-btn")
                reset_btn = gr.Button("Reset", elem_id="reset-btn")

    # Add Modal for chat history
    with Modal(visible=False) as history_modal:
        history_display = gr.Markdown("No chat history available.")

    # Add Modal for PDF selection
    with Modal(visible=False) as pdf_selector_modal:
        gr.Markdown("## Select a PDF")

        with gr.Row():
            pdf_list = gr.Dataframe(
                headers=["Filename", "Size (KB)", "Upload Date"],
                datatype=["str", "number", "str"],
                col_count=(3, "fixed"),
                interactive=False,
                wrap=True,
                show_label=False,
            )

        with gr.Row():
            upload_btn = gr.UploadButton(
                "📁 Upload New PDF", file_types=[".pdf"], elem_id="upload-pdf-btn"
            )

    txt.submit(
        fn=add_text,
        inputs=[chatbot, txt],
        outputs=[chatbot],
        queue=False,
    ).success(
        fn=get_response,
        inputs=[
            chatbot,
            txt,
            upload_btn,  # Changed from btn to upload_btn
            use_semantic_cache,
            use_reranker,
            reranker_type,
            distance_threshold,
            top_k,
            llm_model,
            selected_llm_provider,
            llm_temperature,
            use_chat_history,
            session_state,
        ],
        outputs=[chatbot, txt, feedback_markdown, session_state],
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
            upload_btn,
            use_semantic_cache,
            use_reranker,
            reranker_type,
            distance_threshold,
            top_k,
            llm_model,
            selected_llm_provider,
            llm_temperature,
            use_chat_history,
            session_state,
        ],
        outputs=[chatbot, txt, feedback_markdown, session_state],
    )

    select_pdf_btn.click(
        fn=lambda: (gr.update(visible=True), format_pdf_list(app.search_pdfs())),
        outputs=[pdf_selector_modal, pdf_list],
    )

    pdf_list.select(
        fn=handle_pdf_selection,
        inputs=[pdf_list],
        outputs=[show_pdf, chatbot, feedback_markdown, pdf_selector_modal]
    )

    # First close the modal when user selects a file
    upload_btn.click(fn=lambda: gr.update(visible=False), outputs=pdf_selector_modal)

    upload_btn.upload(
        fn=render_first,
        inputs=[upload_btn, chunk_size, chunking_technique, selected_embedding_model, session_state],
        outputs=[show_pdf, chatbot, session_state],
    ).success(
        fn=lambda: (gr.update(visible=False), format_pdf_list(app.search_pdfs())),
        outputs=[pdf_selector_modal, pdf_list],
    )

    reset_btn.click(
        fn=reset_app,
        inputs=None,
        outputs=[chatbot, show_pdf, txt, feedback_markdown],
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

    use_semantic_cache.change(
        fn=app.update_semantic_cache, inputs=[use_semantic_cache], outputs=[]
    )

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
        outputs=[txt, submit_btn, upload_btn, reset_btn],
    )

    submit_credentials_btn.click(
        fn=update_components_state,
        outputs=[txt, submit_btn, upload_btn, reset_btn],
    )
