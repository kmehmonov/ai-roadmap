from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
import numpy as np
import pickle

# NOTE: This router is not complete yet

router = APIRouter(prefix='/housing')
templates = Jinja2Templates(directory="templates/housing")

@router.get("/")
async def model_page(request: Request):
    return templates.TemplateResponse("main.html", {"request": request, "app_name": "Housing", "prediction": None})



@router.post("/")
async def predict_form(
    request: Request,
    median_income: float = Form(...),
    housing_median_age: float = Form(...),
    total_rooms: float = Form(...),
    total_bedrooms: float = Form(...),
    population: float = Form(...),
    households: float = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...)
):
    input_data = np.array([
        [median_income, housing_median_age, total_rooms, total_bedrooms, population, households, latitude, longitude]
    ])
    print(input_data)
    # prediction = model.predict(input_data)[0]
    prediction = 'Demo prediction'
    return templates.TemplateResponse(
        "main.html", {
            "request": request, 
            "app_name": "Housing", 
            "prediction": prediction
        }
    )


@router.post("/predict")
async def predict_api(input_value: float):
    input_data = np.array([[input_value]])
    # prediction = model.predict(input_data)[0]
    prediction = 'Demo prediction'
    return {"app_name": "Housing", "prediction": prediction}
