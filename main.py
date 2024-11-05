import gradio as gr
from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from demos.workbench import workbench

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")


favicon_path = "static/favicon.ico"

app = gr.mount_gradio_app(app, workbench.demo, workbench.path())


@app.get("/")
async def root():
    return RedirectResponse(url="/workbench")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(favicon_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
