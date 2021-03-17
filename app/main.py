from fastapi import FastAPI

from .routers.v1 import methods

app = FastAPI()
app.include_router(methods.router)

@app.get("/")
async def root():
    return {"message": "HNSW API"}














