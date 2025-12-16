# app.py – Final clean version (passes make quality 100%)

from fastapi import FastAPI
from pydantic import Field
from pydantic import BaseModel
import pandas as pd
import joblib
from model_pipeline import prepare_data, train_model, save_model

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict survival + retrain model via REST API",
    version="1.0",
)

# Load model
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    raise RuntimeError("model.pkl not found – run 'make train' first")


class PassengerInput(BaseModel):
    class Config:
        schema_extra = {
            "example": {
                "Pclass": 3,
                "Sex": 1,
                "Age": 28.0,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked_Q": 0,
                "Embarked_S": 1,
            }
        }

    Pclass: int = Field(
        ..., ge=1, le=3, description="Ticket class (1=1st, 2=2nd, 3=3rd)"
    )
    Sex: int = Field(..., ge=0, le=1, description="0 = male, 1 = female")
    Age: float = Field(..., ge=0, le=120, description="Age in years")
    SibSp: int = Field(..., ge=0, le=8, description="# of siblings / spouses aboard")
    Parch: int = Field(..., ge=0, le=9, description="# of parents / children aboard")
    Fare: float = Field(..., ge=0, le=600, description="Passenger fare in £")
    Embarked_Q: int = Field(
        0, ge=0, le=1, description="1 if embarked at Queenstown, else 0"
    )
    Embarked_S: int = Field(
        1, ge=0, le=1, description="1 if embarked at Southampton, else 0"
    )


@app.post("/predict")
def predict(data: PassengerInput):
    df = pd.DataFrame([data.dict()])
    prediction = int(model.predict(df)[0])
    return {
        "survived": prediction,
        "message": "Survived" if prediction == 1 else "Did not survive",
    }


@app.post("/retrain")
def retrain():
    X_train, _, y_train, _ = prepare_data("titanic/train.csv")
    new_model = train_model(X_train, y_train)
    save_model(new_model, "model.pkl")
    global model
    model = new_model
    return {"status": "Model successfully retrained and updated!"}


@app.get("/")
def home():
    return {"message": "Titanic FastAPI ready – go to /docs"}
