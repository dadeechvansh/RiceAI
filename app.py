from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
from pathlib import Path
from pydantic import BaseModel
from predict import predict_rice

# =====================================================
# Load Project Results
# =====================================================

def load_results():

    result_file = Path("results/results.json")

    if result_file.exists():

        with open(result_file, "r") as file:
            return json.load(file)

    return {}

app = FastAPI(
    title="Rice Variety Classification",
    description="Machine Learning Based Rice Grain Classification",
    version="2.0"
)

# Static Files
app.mount("/static", StaticFiles(directory="static"), name="static")


# Templates
templates = Jinja2Templates(directory="templates")

# =====================================================
# Prediction Request Model
# =====================================================

class PredictionRequest(BaseModel):

    Area: float
    Perimeter: float
    Major_Axis_Length: float
    Minor_Axis_Length: float
    Eccentricity: float
    Convex_Area: float
    Extent: float

# ---------------------------------------------------
# Home Page
# ---------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    results = load_results()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "results": results
        }
    )


# ---------------------------------------------------
# Prediction Page
# ---------------------------------------------------

@app.get("/predict", response_class=HTMLResponse)
async def predictor(request: Request):

    results = load_results()

    return templates.TemplateResponse(
        request=request,
        name="predictor.html",
        context={
            "result": None,
            "results": results
        }
    )


# =====================================================
# API Prediction
# =====================================================

@app.post("/api/predict")
async def api_predict(data: PredictionRequest):

    prediction = predict_rice(
    data.Area,
    data.Perimeter,
    data.Major_Axis_Length,
    data.Minor_Axis_Length,
    data.Eccentricity,
    data.Convex_Area,
    data.Extent
    )

    return JSONResponse(prediction)