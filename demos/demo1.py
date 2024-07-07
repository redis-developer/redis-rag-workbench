import os.path
from typing import Any, Dict, Tuple

import gradio as gr

_ASSET_DIR = os.path.dirname(__file__) + "/assets"

local_prereqs = """
* A ``HUGGING_FACE_HUB_TOKEN`` project secret is required for gated models. See [Tutorial 1](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/README.md#tutorial-1-using-a-local-gpu).
* If using any of the following gated models, verify "You have been granted access to this model" appears on the model card(s):
    * [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)
    * [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
    * [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
    * [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
"""

cloud_info = """
This method uses NVCF API Endpoints from the NVIDIA API Catalog. Select a desired model family and model from the dropdown. You may then query the model using the text input on the left.
"""

cloud_prereqs = """
* A ``NVCF_RUN_KEY`` project secret is required. See the [Quickstart](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/README.md#tutorial-using-a-cloud-endpoint).
    * Generate the key [here](https://build.nvidia.com/mistralai/mistral-7b-instruct-v2) by clicking "Get API Key". Log in with [NGC credentials](https://ngc.nvidia.com/signin).
"""

cloud_trouble = """
* Ensure your NVCF run key is correct and configured properly in the AI Workbench.
"""

nim_info = """
This method uses a [NIM container](https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/llama3-8b-instruct/tags) that you may choose to self-host on your own infra of choice. Check out the NIM [docs](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html) for details. Users can also try 3rd party services supporting the [OpenAI API](https://github.com/ollama/ollama/blob/main/docs/openai.md) like [Ollama](https://github.com/ollama/ollama/blob/main/README.md#building). Input the desired microservice IP, optional port number, and model name under the Remote Microservice option. Then, start conversing using the text input on the left.

For AI Workbench on DOCKER users only, you may also choose to spin up a NIM instance running *locally* on the system by expanding the "Local" Microservice option; ensure any other local GPU processes has been stopped first to avoid issues with memory. The ``llama3-8b-instruct`` NIM container is provided as a default flow. Fetch the desired NIM container, select "Start Microservice", and begin conversing when complete.
"""

nim_prereqs = """
* (Remote) Set up a NIM running on another system ([docs](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html)). Alternatively, you may set up a 3rd party supporting the [OpenAI API](https://github.com/ollama/ollama/blob/main/docs/openai.md) like [Ollama](https://github.com/ollama/ollama/blob/main/README.md#building). Ensure your service is running and reachable. See [Tutorial 2](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/README.md#tutorial-2-using-a-remote-microservice).
* (Local) AI Workbench running on DOCKER is required for the LOCAL NIM option. Read and follow the additional prereqs and configurations in [Tutorial 3](https://github.com/NVIDIA/workbench-example-hybrid-rag/blob/main/README.md#tutorial-3-using-a-local-microservice).
"""

nim_trouble = """
* Send a curl request to your microservice to ensure it is running and reachable. NIM docs [here](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html).
* AI Workbench running on a Docker runtime is required for the LOCAL NIM option. Otherwise, set up the self-hosted NIM to be used remotely.
* If running the local NIM option, ensure you have set up the proper project configurations according to this project's README. Unlike the other inference modes, these are not preconfigured.
* If any other processes are running on the local GPU(s), you may run into memory issues when also running the NIM locally. Stop the other processes.
"""


def load_theme(name: str) -> Tuple[gr.Theme, str]:
    """Load a pre-defined chatui theme.

    :param name: The name of the theme to load.
    :type name: str
    :returns: A tuple containing the Gradio theme and custom CSS.
    :rtype: Tuple[gr.Theme, str]
    """
    theme_json_path = os.path.join(_ASSET_DIR, f"{name}-theme.json")
    theme_css_path = os.path.join(_ASSET_DIR, f"{name}-theme.css")
    return (
        gr.themes.Default().load(theme_json_path),
        open(theme_css_path, encoding="UTF-8").read(),
    )

def chatui_demo_app():
    PATH = "/"
    TITLE = "Redis RAG Workbench"
    OUTPUT_TOKENS = 250
    MAX_DOCS = 5

    ### Load in CSS here for components that need custom styling. ###

    _LOCAL_CSS = """
    #contextbox {
        overflow-y: scroll !important;
        max-height: 400px;
    }

    #params .tabs {
        display: flex;
        flex-direction: column;
        flex-grow: 1;
    }
    #params .tabitem[style="display: block;"] {
        flex-grow: 1;
        display: flex !important;
    }
    #params .gap {
        flex-grow: 1;
    }
    #params .form {
        flex-grow: 1 !important;
    }
    #params .form > :last-child{
        flex-grow: 1;
    }
    #accordion {
    }
    #rag-inputs .svelte-1gfkn6j {
        color: #76b900;
    }
    """
    _LOCAL_CSS = ""

    # Markdown content for the chat UI components
    update_kb_info = """
    <br>
    Upload your text files here. This will embed them in the vector database, and they will persist as potential context for the model until you clear the database. Careful, clearing the database is irreversible!
    """

    local_info = """
    First, select the desired model and quantization level. You can optionally filter the model list by gated vs ungated models. Then load the model. This will either download it or load it from cache. The download may take a few minutes depending on your network.

    Once the model is loaded, start the Inference Server. It takes ~40s to warm up in most cases. Ensure you have enough GPU VRAM to run a model locally or you may see OOM errors when starting the inference server. When the server is started, chat with the model using the text input on the left.
    """

    cloud_info = """
    This method uses NVCF API Endpoints from the NVIDIA API Catalog. Select a desired model family and model from the dropdown. You may then query the model using the text input on the left.
    """

    kui_theme, kui_styles = load_theme("redis")

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles + _LOCAL_CSS) as page:
        gr.HTML("<button class='primary' onclick=\"window.location.href='/demos'\">Back to Demos</button>")

        # create the page header
        gr.Markdown(f"# {TITLE}")

        # Keep track of state we want to persist across user actions
        metrics_history = gr.State({})

        # Build the Chat Application
        with gr.Row(equal_height=True):

            # Left Column will display the chatbot
            with gr.Column(scale=15, min_width=350):

                # Main chatbot panel. Context and Metrics are hidden until toggled
                with gr.Row(equal_height=True):
                    with gr.Column(scale=2, min_width=350):
                        chatbot = gr.Chatbot(show_label=False)

                    context = gr.JSON(
                        scale=1,
                        label="Context",
                        visible=True,
                        elem_id="contextbox",
                    )

                    metrics = gr.JSON(
                        scale=1,
                        label="Metrics",
                        visible=True,
                        elem_id="contextbox",
                    )

                # Render the output sliders to customize the generation output.
                with gr.Tabs(selected=0, visible=False) as out_tabs:
                    with gr.TabItem("Max Tokens", id=0) as max_tokens:
                        num_token_slider = gr.Slider(0, 100, value=200,
                                                     label="The maximum number of tokens that can be generated in the completion.",
                                                     interactive=True)

                    with gr.TabItem("Temperature", id=1) as temperature:
                        temp_slider = gr.Slider(0, 1, value=0.7,
                                                label="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.",
                                                interactive=True)

                    with gr.TabItem("Top P", id=2) as top_p:
                        top_p_slider = gr.Slider(0.001, 0.999, value=0.999,
                                                 label="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.",
                                                 interactive=True)

                    with gr.TabItem("Top K", id=3) as top_k:
                        top_k_slider = gr.Slider(0.001, 0.999, value=0.999,
                                                 label="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.",
                                                 interactive=True)

                    with gr.TabItem("Frequency Penalty", id=4) as freq_pen:
                        freq_pen_slider = gr.Slider(-2, 2, value=0,
                                                    label="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
                                                    interactive=True)

                    with gr.TabItem("Presence Penalty", id=5) as pres_pen:
                        pres_pen_slider = gr.Slider(-2, 2, value=0,
                                                    label="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
                                                    interactive=True)

                    with gr.TabItem("Repetion Penalty", id=6) as rep_pen:
                        rep_pen_slider = gr.Slider(-2, 2, value=0,
                                                    label="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
                                                    interactive=True)

                    with gr.TabItem("Hide LLM Settings", id=7) as hide_out_tools:
                        gr.Markdown("")

                # Hidden button to expand output sliders, if hidden
                out_tabs_show = gr.Button(value="Show LLM Settings", size="sm", visible=True)

                # Render the user input textbox and checkbox to toggle vanilla inference and RAG.
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=200):
                        msg = gr.Textbox(
                            show_label=False,
                            lines=3,
                            placeholder="Enter text and press SUBMIT",
                            container=False,
                            interactive=True,
                        )
                    # with gr.Column(scale=1, min_width=100):
                    #     kb_checkbox = gr.CheckboxGroup(
                    #         ["Toggle to use Vector Database"], label="Vector Database", info="Supply your uploaded documents to the chatbot"
                    #     )

                # Render the row of buttons: submit query, clear history, show metrics and contexts
                with gr.Row():
                    submit_btn = gr.Button(value="Submit", interactive=False)
                    _ = gr.ClearButton([msg, chatbot, metrics, metrics_history], value="Clear history")
                    mtx_show = gr.Button(value="Show Metrics")
                    mtx_hide = gr.Button(value="Hide Metrics", visible=False)
                    ctx_show = gr.Button(value="Show Context")
                    ctx_hide = gr.Button(value="Hide Context", visible=False)

            # Right Column will display the inference and database settings
            with gr.Column(scale=10, min_width=450, visible=True) as settings_column:
                with gr.Tabs(selected=0) as settings_tabs:

                    # First tab item is a button to start the RAG backend and unlock other settings
                    with gr.TabItem("Context", id=0, interactive=True, visible=True) as context:
                        gr.Markdown("<br> ")
                        gr.Markdown("Here we will show the context")
                        reset_context_button = gr.Button(value="Reset Context", variant="primary")
                        gr.Markdown("<br> ")

                    # Second tab item consists of all the inference mode settings
                    with gr.TabItem("Inference Settings", id=1, interactive=True, visible=True) as inf_settings:
                        inference_mode = gr.Radio(["Local System", "Cloud Endpoint", "Self-Hosted Microservice"],
                                                  label="Inference Mode",
                                                  info="Some Info for this",
                                                  value="Cloud Endpoint")

                        # Depending on the selected inference mode, different settings need to get exposed to the user.
                        with gr.Tabs(selected=1) as tabs:

                            # Inference settings for local TGI inference server
                            with gr.TabItem("Local System", id=0, interactive=True, visible=False) as local:
                                with gr.Accordion("Prerequisites", open=True, elem_id="accordion"):
                                    gr.Markdown(local_prereqs)
                                with gr.Accordion("Instructions", open=False, elem_id="accordion"):
                                    gr.Markdown(local_info)
                                with gr.Accordion("Troubleshooting", open=False, elem_id="accordion"):
                                    gr.Markdown("If shit hits the fan")

                                gate_checkbox = gr.CheckboxGroup(
                                    ["Ungated Models", "Gated Models"],
                                    value=["Ungated Models"],
                                    label="Select which models types to show",
                                    interactive = True,
                                    elem_id="rag-inputs")

                                local_model_id = gr.Dropdown(choices = ["nvidia/Llama3-ChatQA-1.5-8B",
                                                                        "microsoft/Phi-3-mini-128k-instruct"],
                                                             value = "nvidia/Llama3-ChatQA-1.5-8B",
                                                             interactive = True,
                                                             label = "Select a model (or input your own).",
                                                             allow_custom_value = True,
                                                             elem_id="rag-inputs")
                                local_model_quantize = gr.Dropdown(choices = ["None",
                                                                              "8-Bit",
                                                                              "4-Bit"],
                                                                   value = "None",
                                                                   interactive = True,
                                                                   label = "Select model quantization.",
                                                                   elem_id="rag-inputs")

                                with gr.Row(equal_height=True):
                                    download_model = gr.Button(value="Load Model", size="sm")
                                    start_local_server = gr.Button(value="Start Server", interactive=False, size="sm")
                                    stop_local_server = gr.Button(value="Stop Server", interactive=False, size="sm")

                            # Inference settings for cloud endpoints inference mode
                            with gr.TabItem("Cloud Endpoint", id=1, interactive=True, visible=True) as cloud:
                                with gr.Accordion("Prerequisites", open=True, elem_id="accordion"):
                                    gr.Markdown(cloud_prereqs)
                                with gr.Accordion("Instructions", open=False, elem_id="accordion"):
                                    gr.Markdown(cloud_info)
                                with gr.Accordion("Troubleshooting", open=False, elem_id="accordion"):
                                    gr.Markdown(cloud_trouble)

                                nvcf_model_family = gr.Dropdown(choices = ["Select",
                                                                           "MistralAI",
                                                                           "Meta",
                                                                           "Google",
                                                                           "Microsoft",
                                                                           "Snowflake",
                                                                           "IBM"],
                                                                value = "Select",
                                                                interactive = True,
                                                                label = "Select a model family.",
                                                                elem_id="rag-inputs")
                                nvcf_model_id = gr.Dropdown(choices = ["Select"],
                                                            value = "Select",
                                                            interactive = True,
                                                            label = "Select a model.",
                                                            visible = False,
                                                            elem_id="rag-inputs")

                            # Inference settings for self-hosted microservice inference mode
                            with gr.TabItem("Self-Hosted Microservice", id=2, interactive=False, visible=False) as microservice:
                                with gr.Accordion("Prerequisites", open=True, elem_id="accordion"):
                                    gr.Markdown(nim_prereqs)
                                with gr.Accordion("Instructions", open=False, elem_id="accordion"):
                                    gr.Markdown(nim_info)
                                with gr.Accordion("Troubleshooting", open=False, elem_id="accordion"):
                                    gr.Markdown(nim_trouble)

                                # User can run a microservice remotely via an endpoint, or as a local inference server.
                                with gr.Tabs(selected=0) as nim_tabs:

                                    # Inference settings for remotely-running microservice
                                    with gr.TabItem("Remote", id=0) as remote_microservice:
                                        remote_nim_msg = gr.Markdown("<br />Enter the details below. Then start chatting!")

                                        with gr.Row(equal_height=True):
                                            nim_model_ip = gr.Textbox(placeholder = "10.123.45.678",
                                                       label = "Microservice Host",
                                                       info = "IP Address running the microservice",
                                                       elem_id="rag-inputs", scale=2)
                                            nim_model_port = gr.Textbox(placeholder = "8000",
                                                       label = "Port",
                                                       info = "Optional, (default: 8000)",
                                                       elem_id="rag-inputs", scale=1)

                                        nim_model_id = gr.Textbox(placeholder = "meta/llama3-8b-instruct",
                                                   label = "Model running in microservice.",
                                                   info = "If none specified, defaults to: meta/llama3-8b-instruct",
                                                   elem_id="rag-inputs")

                                    # Inference settings for locally-running microservice
                                    with gr.TabItem("Local", id=1) as local_microservice:
                                        gr.Markdown("<br />**Important**: For AI Workbench on DOCKER users only. Podman is unsupported!")

                                        nim_local_model_id = gr.Textbox(placeholder = "nvcr.io/nim/meta/llama3-8b-instruct:latest",
                                                   label = "NIM Container Image",
                                                   elem_id="rag-inputs")

                                        with gr.Row(equal_height=True):
                                            prefetch_nim = gr.Button(value="Prefetch NIM", size="sm")
                                            start_local_nim = gr.Button(value="Start Microservice",
                                                                        interactive=(True if os.path.isdir('/mnt/host-home/model-store') else False),
                                                                        size="sm")
                                            stop_local_nim = gr.Button(value="Stop Microservice", interactive=False, size="sm")

                    # Third tab item consists of database and document upload settings
                    with gr.TabItem("Documents", id=2, interactive=True, visible=True) as vdb_settings:

                        gr.Markdown(update_kb_info)

                        file_output = gr.File(interactive=True,
                                              show_label=False,
                                              file_types=["text",
                                                          ".pdf",
                                                          ".html",
                                                          ".doc",
                                                          ".docx",
                                                          ".txt",
                                                          ".odt",
                                                          ".rtf",
                                                          ".tex"],
                                              file_count="multiple")

                        with gr.Row():
                            clear_docs = gr.Button(value="Clear Database", interactive=True, size="sm")

                    # Final tab item consists of option to collapse the settings to reduce clutter on the UI
                    with gr.TabItem("Hide All Settings", id=3, visible=False) as hide_all_settings:
                        gr.Markdown("")

            # Hidden column to be rendered when the user collapses all settings.
            with gr.Column(scale=1, min_width=100, visible=False) as hidden_settings_column:
                show_settings = gr.Button(value="< Expand", size="sm")

            def _toggle_hide_out_tools() -> Dict[gr.component, Dict[Any, Any]]:
                """ Event listener to hide output toolbar from the user. """
                return {
                    out_tabs: gr.update(visible=False, selected=0),
                    out_tabs_show: gr.update(visible=True),
                }

            hide_out_tools.select(_toggle_hide_out_tools, None, [out_tabs, out_tabs_show])

            def _toggle_show_out_tools() -> Dict[gr.component, Dict[Any, Any]]:
                """ Event listener to expand output toolbar for the user. """
                return {
                    out_tabs: gr.update(visible=True),
                    out_tabs_show: gr.update(visible=False),
                }

            out_tabs_show.click(_toggle_show_out_tools, None, [out_tabs, out_tabs_show])

    return page

demo = chatui_demo_app()

def app_title():
    return "Redis RAG Workbench Prototype"

def path():
    return "/rag_workbench"
