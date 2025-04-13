from fastapi.testclient import TestClient
from main import app

# Initialize the TestClient for the FastAPI app
client = TestClient(app)


def test_get_root():
    """ Test the GET / root endpoint """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the FastAPI ML Inference Service!"}


def test_post_predict_less_than_50k():
    """ Test POST /predict endpoint with a prediction of '<=50K' """
    test_data = {
        "age": 30,
        "workclass": "Private",
        "fnlgt": 100000,
        "education": "Bachelors",
        "education-num": 13,  # This should stay as is because the field is defined with a hyphen
        "marital-status": "Never-married",  # Corrected: Use hyphen here
        "occupation": "Tech-support",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,  # Corrected: Use hyphen here
        "capital-loss": 0,  # Corrected: Use hyphen here
        "hours-per-week": 40,  # Corrected: Use hyphen here
        "native-country": "United-States"  # Corrected: Use hyphen here
    }
    response = client.post("/predict/", json=test_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": "<=50K"}


def test_post_predict_greater_than_50k():
    """ Test POST /predict endpoint with a prediction of '>50K' """
    test_data = {
        "age": 40,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 100000,
        "education": "Masters",
        "education-num": 13,  # This should stay as is because the field is defined with a hyphen
        "marital-status": "Married-civ-spouse",  # Corrected: Use hyphen here
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5000,  # Corrected: Use hyphen here
        "capital-loss": 0,  # Corrected: Use hyphen here
        "hours-per-week": 50,  # Corrected: Use hyphen here
        "native-country": "United-States"  # Corrected: Use hyphen here
    }
    response = client.post("/predict/", json=test_data)
    assert response.status_code == 200
    assert response.json() == {"prediction": ">50K"}
