from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from image_gen import generate_image
import io
from PIL import Image
import base64

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-image/")
async def create_image(prompt: str = Form(...)):
    image = generate_image(prompt)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_str = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    return {"image": img_str}

@app.get("/image")
async def get_image():
    return {"message": "Image generation requires a POST to /generate-image/"}