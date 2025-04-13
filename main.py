# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from starter.ml.data import process_data
import pandas as pd

app = FastAPI()

# Load the pre-trained model
model = joblib.load('model/model.pkl')

# Define a Pydantic model for the POST request body


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        def alias_generator(x):
            return x.replace('_', '-')
        validate_by_name = True

        json_schema_extra = {
            "example": {
                "age": 40,
                "workclass": "Private",
                "fnlgt": 234721,
                "education": "HS-grad",
                "education-num": 9,
                "marital-status": "Married-civ-spouse",  # Note the hyphen here
                "occupation": "Tech-support",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,  # Hyphenated format
                "capital-loss": 0,  # Hyphenated format
                "hours-per-week": 40,  # Hyphenated format
                "native-country": "United-States"  # Hyphenated format
            }
        }


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI ML Inference Service!"}


# POST method for model inference
@app.post("/predict/")
def predict(data: InputData):
    # Convert the Pydantic model to a pandas DataFrame
    input_data = pd.DataFrame([data.model_dump(by_alias=True)])

    # Define the categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    encoder = joblib.load("model/encoder.pkl")
    lb = joblib.load("model/lb.pkl")
    # Process the data
    X, _, _, _ = process_data(input_data, categorical_features=cat_features,
                              label=None, training=False, encoder=encoder, lb=lb)

    # Make prediction
    prediction = model.predict(X)
    prediction_label = lb.inverse_transform(prediction)[0]

    # Return the prediction
    return {"prediction": prediction_label}
