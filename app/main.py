import shutil
import timeit
from fastapi import FastAPI, APIRouter
from fastapi import Request
from fastapi import UploadFile
from fastapi.templating import Jinja2Templates

from model_functions import CNNModel

app = FastAPI(title="Title")
api_router = APIRouter()

templates = Jinja2Templates(directory="templates")

cnn_model: CNNModel


@api_router.on_event("startup")
def init_cnn_model():
    global cnn_model
    cnn_model = CNNModel(filepath="ml_models/dogvscat.h5")


@api_router.get("/")
def root(request: Request):
    ...
    return {"message": "/"}


@api_router.post("/uploadfile/")
def create_upload_file(file: UploadFile):
    start = timeit.default_timer()

    with open(f"{file.filename}", 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    res = cnn_model.predict_image(image_path=file.filename)

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    return {"filename": file.filename,
            "result": res}


app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8001, log_level="debug", reload=True)
