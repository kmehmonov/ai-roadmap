import json
import joblib
import csv
from dataclasses import dataclass

import pandas as pd
import numpy as np
from fastapi import APIRouter, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.exceptions import HTTPException
from fastapi.templating import Jinja2Templates
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

preprocessor: Pipeline = joblib.load('models/iris/preprocessor.joblib')
log_clf: LogisticRegression = joblib.load('models/iris/log_clf.joblib')

with open('models/iris/class_names.json') as f:
    class_names = json.load(f)

# svm_clf: SVC = load_model("log_reg.pkl")
# dt_clf: DecisionTreeClassifier = load_model("dt_clf.pkl")

# input_pipe: Pipeline = load_model("input_pipe.pkl")
# le: LabelEncoder = load_model("le.pkl")

router = APIRouter(prefix='/iris')
templates = Jinja2Templates(directory="templates/iris")


@dataclass
class Feature:
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    def as_X(self):
        X = np.array([
                [self.sepal_length, self.sepal_width,self.petal_length,self.petal_width]
        ])

        return X


@router.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse(request=request, name="main.html")


@router.post("/predict")
async def predict(features: Feature = Body(), model_type: str = Body()):
    X = features.as_X()
    X = preprocessor.transform(X)
    class_id = log_clf.predict(X)[0].item()
    proba = log_clf.predict_proba(X)[0][class_id]
    proba = round(proba, 2)
    class_name = class_names[str(class_id)]

    # # FIXME: ugly code
    # if model_type == "log_reg":
    #     out = log_reg.predict(X)
    # elif model_type == "svm_clf":
    #     out = svm_clf.predict(X)
    # elif model_type == 'dt_clf':
    #     out = dt_clf.predict(X)
    # else:
    #     raise HTTPException(status_code=404, detail='Unknown model type')

    # iris_type = le.inverse_transform(out)[0]

    # print(f"{model_type=}")
    # print(f"{iris_type=}")

    # # FIXME: hardcoded urls
    # match iris_type:
    #     case "Iris-setosa":
    #         img_url = "http://127.0.0.1:8000/static/img/iris-setosa1.jpg"
    #     case "Iris-versicolor":
    #         img_url = "http://127.0.0.1:8000/static/img/iris-versicolor1.jpg"
    #     case "Iris-virginica":
    #         img_url = "http://127.0.0.1:8000/static/img/iris-virginica1.jpg"
    #     case _:
    #         img_url = None

    return {"features": features, "iris_type": class_name, "probability": proba}


@router.get("/data")
async def data(request: Request):
    columns = [
        "Id",
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm",
        "Species",
    ]
    species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    context = {"columns": columns, "species": species}
    return templates.TemplateResponse(
        request=request, name="data.html", context=context
    )


@router.get("/api/data")
async def api_data():
    with open("ai/data/Iris.csv", "rt", encoding="utf8") as f:
        dict_reader = csv.DictReader(f)
        data = list(dict_reader)

    return {"data": data, "columns": dict_reader.fieldnames}


@router.get("/eda")
async def eda(request: Request, x: str = None, y: str = None):
    xval = "SepalLengthCm"
    yval = "SepalWidthCm"
    if x and y:
        xval = x
        yval = y

    df = pd.read_csv("ai/data/Iris.csv")
    fig1 = px.scatter(
        df, x=xval, y=yval, color="Species", title="Iris Dataset 2D Scatter Plot"
    )
    fig1_json = json.dumps(obj=fig1, cls=PlotlyJSONEncoder)

    fig2 = px.scatter_3d(
        df,
        x="SepalLengthCm",
        y="SepalWidthCm",
        z="PetalWidthCm",
        color="Species",
        title="Iris dataset 3D Scatter Plot",
    )
    fig2_json = json.dumps(fig2, cls=PlotlyJSONEncoder)

    context = {
        "fig1_json": fig1_json,
        "fig2_json": fig2_json,
    }
    return templates.TemplateResponse(request=request, name="eda.html", context=context)
