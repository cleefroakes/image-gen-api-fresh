from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from image_gen import generate_image
from fastapi import Form
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

class ImageRequest(BaseModel):
    prompt: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-image/")
async def create_image(prompt: str = Form(...)):
    image = generate_image(prompt)
    import io
    from PIL import Image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return {"image": img_byte_arr.decode('latin1') if isinstance(img_byte_arr, bytes) else img_byte_arr}

@app.get("/image")
async def get_image():
    return {"message": "Image generation requires a POST to /generate-image/"}