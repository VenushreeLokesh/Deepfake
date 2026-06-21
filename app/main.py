import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn

from app.inference import gan_model, celeb_model, run_inference

# ── App state ─────────────────────────────────────────────────────────────────
model_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Both models already loaded globally in inference.py — just store references
    model_store['gan_model']   = gan_model
    model_store['celeb_model'] = celeb_model
    print("✅ Models ready")
    yield
    model_store.clear()

app = FastAPI(title="Deepfake Localisation API", lifespan=lifespan)

# ── Serve static files (index.html) ───────────────────────────────────────────
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPG/PNG images are supported.")

    image_bytes = await file.read()

    try:
        result = run_inference(image_bytes)  # no model argument — handled internally
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result_image_b64 = base64.b64encode(result['result_image']).decode('utf-8')

    return JSONResponse({
        'label':              result['label'],
        'confidence':         result['confidence'],
        'localisation_score': result['localisation_score'],
        'weak_localisation':  result['weak_localisation'],
        'result_image':       result_image_b64,
    })

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)