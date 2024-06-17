import gradio as gr

def demo2_app():
    def demo2_fn(text):
        return f"Demo 2: {text}"

    with gr.Blocks() as demo2:
        gr.HTML("<button onclick=\"window.location.href='/demos'\">Back to Demos</button>")
        input_text = gr.Textbox(label="Enter something", placeholder="Type here...")
        output_text = gr.Text(label="Output")
        input_text.change(fn=demo2_fn, inputs=input_text, outputs=output_text)

    return demo2

demo = demo2_app()
