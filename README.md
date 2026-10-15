\# RabTech ML Inference REST API



\## Overview

This project provides a machine learning prediction service using FastAPI and a trained Scikit-learn pipeline.



\## Features

\- REST API for model predictions

\- Health check endpoint

\- Input validation

\- Automated tests

\- Docker support



\## API Endpoints

\- GET / — Check API status

\- GET /health — Check model status

\- POST /predict — Generate a prediction



\## Run Locally

Install dependencies:



```bash

pip install -r requirements.txt

```



Start the API:



```bash

uvicorn app:app --reload

```



Open http://127.0.0.1:8000/docs to test the API.



\## Run Tests

```bash

python -m pytest -v

```



\## Docker

Build and run the Docker image:



```bash

docker build -t rabtech-ml-api .

docker run -p 8000:8000 rabtech-ml-api

```



\## Model

The API loads the trained `champion\_model.joblib` pipeline.



\## Disclaimer

Predictions are for demonstration and educational purposes.



