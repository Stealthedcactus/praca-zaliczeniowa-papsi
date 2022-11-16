import pickle

import numpy as np


# Może być pobrane z bazy danych
available_models = {"DecisionTreeClassifier": "ml_models/model_trained-ver002.pkl"}


def load_model(model_choice: str):
    try:
        with open(available_models[model_choice], 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        print(f"problem with opening file, \n{e}")
    else:
        print(f'Model loaded!!!!!')
        return model


def model_predict(string_, model):
    string = string_.lower()
    temp = np.array([[int(len(string)), int(string[-1] == 'a')]])
    return str(model.predict(temp)[0])
