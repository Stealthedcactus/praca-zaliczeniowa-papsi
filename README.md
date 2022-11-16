###1 Połaczenie adminerem z bazą danych:

- docker compose up --build
- http://127.0.0.1:8001/

- POST on http://127.0.0.1:8001/predict 

  - Json: {
  "model_choice": "DecisionTreeClassifier",
  "user_input": "UserInput"
}