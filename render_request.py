import requests

# Replace this with your actual deployed Render URL
url = "https://your-app-name.onrender.com/predict/"

# Define the input data using hyphenated keys to match the API schema
input_data = {
    "age": 40,
    "workclass": "Private",
    "fnlgt": 234721,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Tech-support",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

# Send the POST request
response = requests.post(url, json=input_data)

# Print the response
print("Status code:", response.status_code)
print("Response:", response.json())
