import os
import gradio as gr
from typing import Tuple

_ASSET_DIR = os.path.dirname(__file__) + "/../demos/assets"

def load_theme(name: str) -> Tuple[gr.Theme, str]:
    """Load a pre-defined gradio theme.

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