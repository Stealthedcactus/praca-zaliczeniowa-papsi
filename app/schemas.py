from pydantic import BaseModel

"""
W tym pliku znajdują się pydanticowe klasy,
odpowiadające za walidację danych

Definiowane są tu struktury odpowiedzi API
"""


class Predict(BaseModel):
    model_choice: str
    user_input: str


class PredictResponse(BaseModel):
    response: str

