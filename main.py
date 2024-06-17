import os
import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from demos import demo1, demo2, demo3

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Custom CSS for centering the list and making it scrollable
css = """
html, body {
    height: 100%;
    margin: 0;
    display: flex;
    justify-content: center;
    align-items: center;
    background-color: #f5f5f5;
}

.centered-container {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 100%;
    width: 100%;
}

.scrollable-list {
    max-height: 80vh;
    overflow-y: auto;
    width: 300px;
    text-align: center;
}
.scrollable-list button {
    width: 100%;
    padding: 10px;
    margin: 5px 0;
    border: none;
    background-color: #3498db;
    color: white;
    cursor: pointer;
}
.scrollable-list button:hover {
    background-color: #2980b9;
}
"""

# Create navigation Gradio app
def navigation():
    with gr.Blocks(css=css) as demo_list:
        with gr.Column(elem_id="centered-container"):
            gr.Markdown("## Demo List")
            with gr.Column(elem_id="scrollable-list"):
                demos = ["demo1", "demo2", "demo3"]
                for demo in demos:
                    gr.HTML(f"<button onclick=\"window.location.href='/{demo}'\">{demo}</button>")

    return demo_list

demo_list_app = navigation()
app.mount("/demos", gr.mount_gradio_app(app, demo_list_app, path="/demos"))

# Mount individual Gradio demo apps
app.mount("/demo1", gr.mount_gradio_app(app, demo1.demo, path="/demo1"))
app.mount("/demo2", gr.mount_gradio_app(app, demo2.demo, path="/demo2"))
app.mount("/demo3", gr.mount_gradio_app(app, demo3.demo, path="/demo3"))

@app.get("/")
async def root():
    return RedirectResponse(url="/demos")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
