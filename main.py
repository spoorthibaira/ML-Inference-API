
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import time

app = FastAPI(
    title="Iris Flower Prediction API",
    description="Machine Learning Inference API using Random Forest",
    version="1.0.0"
)

# Load trained model
model = joblib.load("iris_model.pkl")
flower_names = joblib.load("flower_names.pkl")


# Input validation
class FlowerInput(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)


@app.get("/")
def home():
    return {"message": "Iris Prediction API is running!"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": "1.0.0"}


@app.post("/predict")
def predict(data: FlowerInput):
    start_time = time.perf_counter()

    input_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(input_data)[0]

    latency = (time.perf_counter() - start_time) * 1000

    return {
        "prediction": str(flower_names[prediction]),
        "class_id": int(prediction),
        "model_version": "1.0.0",
        "latency_ms": round(latency, 3)
    }
