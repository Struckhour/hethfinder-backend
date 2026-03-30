from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.services.plotting import generate_sine_plot, generate_cosine_plot, generate_scatter_plot

router = APIRouter()

@router.get("/plot/sine")
def get_sine_plot():
    img = generate_sine_plot()
    return StreamingResponse(img, media_type="image/png")

@router.get("/plot/cosine")
def get_cosine_plot():
    img = generate_cosine_plot()
    return StreamingResponse(img, media_type="image/png")

@router.get("/plot/scatter")
def get_scatter_plot():
    img = generate_scatter_plot()
    return StreamingResponse(img, media_type="image/png")