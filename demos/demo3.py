import gradio as gr
import time
import os.path
from typing import Tuple

_ASSET_DIR = os.path.dirname(__file__) + "/assets"

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

def demo3_app():
    _LOCAL_CSS = """
    """

    def demo3_fn(text):
        return f"Demo 3: {text}"

    redis_theme, redis_styles = load_theme("redis")

    with gr.Blocks(theme=redis_theme, css=redis_styles + _LOCAL_CSS) as demo3:
        gr.HTML("<button class='primary' onclick=\"window.location.href='/demos'\">Back to Demos</button>")
        dark_mode_btn = gr.Button("Toggle Light/Dark Mode", variant="primary", size="sm")

        with gr.Column(scale=6, elem_id="app"):
            with gr.Column(variant="panel"):
                gr.Markdown(
                    """
                    # Theme Builder
                    Welcome to the theme builder. The left panel is where you create the theme. The different aspects of the theme are broken down into different tabs. Here's how to navigate them:
                    1. First, set the "Source Theme". This will set the default values that you can override.
                    2. Set the "Core Colors", "Core Sizing" and "Core Fonts". These are the core variables that are used to build the rest of the theme.
                    3. The rest of the tabs set specific CSS theme variables. These control finer aspects of the UI. Within these theme variables, you can reference the core variables and other theme variables using the variable name preceded by an asterisk, e.g. `*primary_50` or `*body_text_color`. Clear the dropdown to set a custom value.
                    4. Once you have finished your theme, click on "View Code" below to see how you can integrate the theme into your app. You can also click on "Upload to Hub" to upload your theme to the Hugging Face Hub, where others can download and use your theme.
                    """
                )
                with gr.Accordion("View Code", open=False):
                    output_code = gr.Code(language="python")
                with gr.Accordion("Upload to Hub", open=False):
                    gr.Markdown(
                        "You can save your theme on the Hugging Face Hub. HF API write token can be found [here](https://huggingface.co/settings/tokens)."
                    )
                    with gr.Row():
                        theme_name = gr.Textbox(label="Theme Name")
                        theme_hf_token = gr.Textbox(label="Hugging Face Write Token")
                        theme_version = gr.Textbox(
                            label="Version",
                            placeholder="Leave blank to automatically update version.",
                        )
                    upload_to_hub_btn = gr.Button("Upload to Hub")
                    theme_upload_status = gr.Markdown(visible=False)

                gr.Markdown("Below this panel is a dummy app to demo your theme.")

            name = gr.Textbox(
                label="Name",
                info="Full name, including middle name. No special characters.",
                placeholder="John Doe",
                value="John Doe",
                interactive=True,
            )

            with gr.Row():
                slider1 = gr.Slider(label="Slider 1")
                slider2 = gr.Slider(label="Slider 2")
            gr.CheckboxGroup(["A", "B", "C"], label="Checkbox Group")

            with gr.Row():
                with gr.Column(variant="panel", scale=1):
                    gr.Markdown("## Panel 1")
                    radio = gr.Radio(
                        ["A", "B", "C"],
                        label="Radio",
                        info="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
                    )
                    drop = gr.Dropdown(
                        ["Option 1", "Option 2", "Option 3"], show_label=False
                    )
                    drop_2 = gr.Dropdown(
                        ["Option A", "Option B", "Option C"],
                        multiselect=True,
                        value=["Option A"],
                        label="Dropdown",
                        interactive=True,
                    )
                    check = gr.Checkbox(label="Go")
                with gr.Column(variant="panel", scale=2):
                    img = gr.Image(
                        "https://gradio-static-files.s3.us-west-2.amazonaws.com/header-image.jpg",
                        label="Image",
                        height=320,
                    )
                    with gr.Row():
                        go_btn = gr.Button("Go", variant="primary")
                        clear_btn = gr.Button("Clear", variant="secondary")

                        def go(*_args):
                            time.sleep(3)
                            return "https://gradio-static-files.s3.us-west-2.amazonaws.com/header-image.jpg"

                        go_btn.click(
                            go,
                            [radio, drop, drop_2, check, name],
                            img,
                            show_api=False,
                        )

                        def clear():
                            time.sleep(0.2)

                        clear_btn.click(clear, None, img)

                    with gr.Row():
                        btn1 = gr.Button("Button 1", size="sm")
                        btn2 = gr.UploadButton(size="sm")
                        stop_btn = gr.Button("Stop", variant="stop", size="sm")

            gr.Examples(
                examples=[
                    [
                        "A",
                        "Option 1",
                        ["Option B"],
                        True,
                    ],
                    [
                        "B",
                        "Option 2",
                        ["Option B", "Option C"],
                        False,
                    ],
                ],
                inputs=[radio, drop, drop_2, check],
                label="Examples",
            )

            with gr.Row():
                gr.Dataframe(value=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], label="Dataframe")
                gr.JSON(
                    value={"a": 1, "b": 2, "c": {"test": "a", "test2": [1, 2, 3]}},
                    label="JSON",
                )
                gr.Label(value={"cat": 0.7, "dog": 0.2, "fish": 0.1})
                gr.File()
            with gr.Row():
                gr.ColorPicker()
                gr.Video(
                    "https://gradio-static-files.s3.us-west-2.amazonaws.com/world.mp4"
                )
                gr.Gallery(
                    [
                        (
                            "https://gradio-static-files.s3.us-west-2.amazonaws.com/lion.jpg",
                            "lion",
                        ),
                        (
                            "https://gradio-static-files.s3.us-west-2.amazonaws.com/logo.png",
                            "logo",
                        ),
                        (
                            "https://gradio-static-files.s3.us-west-2.amazonaws.com/tower.jpg",
                            "tower",
                        ),
                    ],
                    height="200px",
                    columns=2,
                )

            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot([("Hello", "Hi")], label="Chatbot")
                    chat_btn = gr.Button("Add messages")

                    chat_btn.click(
                        lambda history: history
                        + [["How are you?", "I am good."]]
                        + (time.sleep(2) or []),
                        chatbot,
                        chatbot,
                        show_api=False,
                    )
                with gr.Column(scale=1):
                    with gr.Accordion("Advanced Settings"):
                        gr.Markdown("Hello")
                        gr.Number(label="Chatbot control 1")
                        gr.Number(label="Chatbot control 2")
                        gr.Number(label="Chatbot control 3")

            dark_mode_btn.click(
                None,
                None,
                None,
                js="""() => {
                if (document.querySelectorAll('.dark').length) {
                    document.querySelectorAll('.dark').forEach(el => el.classList.remove('dark'));
                    console.log("Disabling Dark Mode");
                } else {
                    document.querySelector('body').classList.add('dark');
                    console.log("Enabling Dark Mode");
                }
            }""",
                show_api=False,
            )
    return demo3

demo = demo3_app()

def app_title():
    return "Redis Gradio Theme Tester"

def path():
    return "/theme_tester"
