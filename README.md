
# Machine Learning Inference API Using FastAPI

## 1. Project Overview

This project demonstrates how to build a Machine Learning Inference API using FastAPI.

It uses the Iris dataset and a Random Forest Classifier to predict the species of a flower based on four measurements.

## 2. Technologies Used

- Python 3.13
- FastAPI
- Scikit-learn
- Pandas
- Joblib
- Uvicorn

## 3. Machine Learning Model

The Iris dataset contains 150 flower samples belonging to three species:

- Setosa
- Versicolor
- Virginica

A Random Forest Classifier is trained using 80% of the dataset and tested using the remaining 20%.

The model achieved 1.0 accuracy on the test set.

## 4. Project Structure

```text
ML-Inference-API/
├── main.py
├── train_model.py
├── iris_model.pkl
├── flower_names.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
```

## 5. Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ML-Inference-API
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 6. Train the Model

```bash
python train_model.py
```

## 7. Run the API

```bash
python -m uvicorn main:app --reload
```

Open Swagger documentation:

http://127.0.0.1:8000/docs

## 8. API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | / | Home |
| GET | /health | Health check |
| POST | /predict | Predict flower species |

## 9. Sample Input

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

## 10. Sample Prediction

```json
{
  "prediction": "setosa",
  "class_id": 0,
  "model_version": "1.0.0",
  "latency_ms": 0.5
}
```

The latency shown above is an example; actual latency may vary.

## 11. Input Validation

The API uses Pydantic to validate input measurements.

Invalid values, such as negative measurements, are rejected with a 422 response.

## 12. Model Versioning

The API identifies its model using version 1.0.0.

## 13. Conclusion

This project demonstrates how a trained Machine Learning model can be integrated with FastAPI to provide real-time predictions through an API.
