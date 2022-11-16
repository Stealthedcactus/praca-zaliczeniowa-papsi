from fastapi import FastAPI, APIRouter
from fastapi import Request
from fastapi.templating import Jinja2Templates

from model_functions import load_model, model_predict
from schemas import Predict, PredictResponse

app = FastAPI(title="Title")
api_router = APIRouter()

templates = Jinja2Templates(directory="templates")


@api_router.get("/")
def root(request: Request):
    ...
    return templates.TemplateResponse("index.html", {"request": request})


@api_router.post("/predict", response_model=PredictResponse)
def predict(*, prediction_in: Predict):
    model = load_model(model_choice=prediction_in.model_choice)
    model_prediction = model_predict(string_=prediction_in.user_input, model=model)

    return PredictResponse(response=model_prediction)


app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8001, log_level="debug", reload=True)
