from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from web.router import router

app = FastAPI()

app.mount('/static', StaticFiles(directory='web/static'), name='static')
app.include_router(router)
