import gradio as gr
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from demos.workbench import workbench

app = FastAPI()  # lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")

favicon_path = "static/favicon.ico"


@app.get("/")
async def root():
    return RedirectResponse(url="/workbench")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


workbench.initialize()
app = gr.mount_gradio_app(app, workbench.ui(), workbench.path())
