import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from demos.chat_with_pdf import chat_with_pdf

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Custom CSS for centering the list and making it scrollable
# css = """
# html, body {
#     height: 100%;
#     margin: 0;
#     display: flex;
#     justify-content: center;
#     align-items: center;
#     background-color: #f5f5f5;
# }

# .centered-container {
#     display: flex;
#     flex-direction: column;
#     justify-content: center;
#     align-items: center;
#     height: 100%;
#     width: 100%;
# }

# .scrollable-list {
#     max-height: 80vh;
#     overflow-y: auto;
#     width: 300px;
#     text-align: center;
# }
# .scrollable-list button {
#     width: 100%;
#     padding: 10px;
#     margin: 5px 0;
#     border: none;
#     background-color: #3498db;
#     color: white;
#     cursor: pointer;
# }
# .scrollable-list button:hover {
#     background-color: #2980b9;
# }
# """

app.mount(
    chat_with_pdf.path(),
    gr.mount_gradio_app(app, chat_with_pdf.demo, chat_with_pdf.path()),
)


@app.get("/")
async def root():
    return RedirectResponse(url="/chat_with_pdf")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
