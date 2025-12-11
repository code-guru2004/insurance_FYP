from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ----- Input Schema -----
class InsuranceRequest(BaseModel):
    age: int
    sex: int          # male=0, female=1
    bmi: float
    children: int
    smoker: int       # yes=0, no=1
    region: int       # southeast=0, southwest=1, northeast=2, northwest=3

@app.post("/predict")
def predict_insurance(data: InsuranceRequest):
    
    # Convert input into required numpy format
    input_data = np.array([
        [
            data.age,
            data.sex,
            data.bmi,
            data.children,
            data.smoker,
            data.region
        ]
    ])

    # Model prediction
    prediction = model.predict(input_data)

    return {
        "prediction": float(prediction[0]),
        "currency": "USD"
    }
