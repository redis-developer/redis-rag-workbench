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

def demo2_app():
    _LOCAL_CSS = """
    """

    def demo2_fn(text):
        return f"Demo 2: {text}"

    redis_theme, redis_styles = load_theme("redis")

    with gr.Blocks(theme=redis_theme, css=redis_styles + _LOCAL_CSS) as demo2:
        gr.HTML("<button class='primary' onclick=\"window.location.href='/demos'\">Back to Demos</button>")
        input_text = gr.Textbox(label="Enter something", placeholder="Type here...")
        output_text = gr.Text(label="Output")

        textbox = gr.Textbox(label="Name")
        input_text.change(fn=demo2_fn, inputs=input_text, outputs=output_text)

        slider = gr.Slider(label="Count", minimum=0, maximum=100, step=1)
        with gr.Row():
            button = gr.Button("Submit", variant="primary")
            clear = gr.Button("Clear")
        output = gr.Textbox(label="Output")

        def repeat(name, count):
            time.sleep(3)
            return name * count

        button.click(repeat, [textbox, slider], output)


    return demo2

demo = demo2_app()

def app_title():
    return "Gradio Test Page"

def path():
    return "/test_page"
