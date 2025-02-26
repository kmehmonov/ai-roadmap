from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from routers.housing import housing_router
from routers.iris import iris_router

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(housing_router, tags=["Housing"])
app.include_router(iris_router, tags=["Iris"])

REGISTERED_APPS = [
    {"name": "Housing", "url": "/housing"},
    {"name": "Iris", "url": "/iris"}
]

@app.get("/", tags=["Home"])
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "apps": REGISTERED_APPS})

# Run using: uvicorn main:app --reload
