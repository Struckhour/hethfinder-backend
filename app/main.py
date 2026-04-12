from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# relative import from the routes package
from .routes import spectrogram
from .routes import predict


app = FastAPI()

# ---- CORS setup ----
origins = [
    "http://localhost:5173",  # Svelte dev server
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include router
app.include_router(spectrogram.router)
app.include_router(predict.router)