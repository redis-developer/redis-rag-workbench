import gradio as gr

def demo1_app():
    def demo1_fn(text):
        return f"Demo 1: {text}"

    with gr.Blocks() as demo1:
        gr.HTML("<button onclick=\"window.location.href='/demos'\">Back to Demos</button>")
        input_text = gr.Textbox(label="Enter something", placeholder="Type here...")
        output_text = gr.Text(label="Output")
        input_text.change(fn=demo1_fn, inputs=input_text, outputs=output_text)

    return demo1

demo = demo1_app()
