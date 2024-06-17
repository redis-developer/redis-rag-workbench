import gradio as gr

def demo3_app():
    def demo3_fn(text):
        return f"Demo 3: {text}"

    with gr.Blocks() as demo3:
        gr.HTML("<button onclick=\"window.location.href='/demos'\">Back to Demos</button>")
        input_text = gr.Textbox(label="Enter something", placeholder="Type here...")
        output_text = gr.Text(label="Output")
        input_text.change(fn=demo3_fn, inputs=input_text, outputs=output_text)

    return demo3

demo = demo3_app()
