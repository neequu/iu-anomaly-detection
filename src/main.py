from fastapi import FastAPI

from .api import api_router

app = FastAPI()
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
