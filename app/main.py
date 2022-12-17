import base64
import shutil
import timeit
from datetime import datetime

from fastapi import FastAPI, APIRouter
from fastapi import UploadFile, File, Request
from fastapi.templating import Jinja2Templates

from model_functions import CNNModel, result_translate

app = FastAPI(title="Title")
api_router = APIRouter()
templates = Jinja2Templates(directory="templates")
cnn_model: CNNModel


@api_router.on_event("startup")
def init_cnn_model():
    global cnn_model
    cnn_model = CNNModel(filepath="ml_models/dogvscat.h5",
                         test_image_path="test.jpg")


@api_router.get("/")
def main(request: Request):
    # Load image & convert to base64
    with open("test.jpg", 'rb') as file:
        image = base64.b64encode(file.read()).decode("utf-8")

    return templates.TemplateResponse("index.html", {"request": request, "example_file": image})


@api_router.post("/upload2")
def upload(file: UploadFile = File(...)):
    start = timeit.default_timer()

    with open(f"tested_images/{file.filename}", 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    res = cnn_model.predict_image(image_path=f"tested_images/{file.filename}")
    res = round(float(res), 5)

    stop = timeit.default_timer()
    print(f'Time: {stop - start / 1000} ms')
    print(f'{result_translate(res)}')

    time_now = datetime.now().time()
    time_now = f"{time_now.hour}:{time_now.minute}:{time_now.second} "

    return {"message": f"Successfuly uploaded {file.filename}",
            "result": result_translate(res),
            "time_stamp": time_now}


app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, log_level="debug", reload=True)
