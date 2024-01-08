import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, Response
from autogluon.tabular import TabularPredictor

"""
This module sets up a Flask web server for serving predictions from a trained AutoGluon model.
The server exposes two endpoints:

1. `/ping`: A health check endpoint that returns "pong" to indicate that the server is running.
2. `/predict`: An endpoint that accepts POST requests with JSON content. Each request should contain one
   or more instances for which predictions are desired. The endpoint returns the predictions and
   associated probabilities in a JSON response.

The server expects an environment variable `MODEL_PATH` that points to the directory where the AutoGluon
model artifacts are stored. If `MODEL_PATH` is not provided, it defaults to '/app/model'.

Usage:
    To build the Docker image, navigate to the directory containing your Dockerfile and execute:
    ```
    docker build -f model_oss/autogluon/dockerfile/serve.Dockerfile -t your_image_name .
    ```

    To run the Docker container after building the image, use:
    ```
    docker run -p 8501:8501 -v local_model_path:/autogluon/models your_image_name
    ```

    Replace the following placeholders:
    - your_image_name: The name you want to assign to your Docker image.
    - local_model_path: The local file path to the trained AutoGluon model artifacts.


    To send a prediction request, use the `curl` command or any HTTP client with the following JSON payload format:
    ```
    {
      "instances": [
        {
          "feature1": value1,
          "feature2": value2,
          ...
        },
        ...
      ]
    }
    ```

    For example:
    ```
    curl -d '{"instances": [{"feature1": value1, "feature2": value2}]}' \
         -H "Content-Type: application/json" \
         -X POST http://localhost:8501/predict
    ```
"""

app = Flask(__name__)

# Assuming the model artifacts are in a directory called /app/model
MODEL_DIR = os.getenv('model_path', '/autogluon/models')

# Load the predictor at startup
predictor = TabularPredictor.load(MODEL_DIR)

@app.route('/ping', methods=['GET'])
def ping():
    """Health check route"""
    return Response("pong", status=200)


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction route"""
    try:
        # Extract JSON content from the POST request
        data = request.get_json(force=True)
        instances = data.get("instances", [])

        # Convert instances to DataFrame
        df_to_predict = pd.DataFrame(instances)

        # Perform prediction
        predictions = predictor.predict(df_to_predict).tolist()

        response = {
            "predictions": predictions
        }

        return Response(json.dumps(response), status=200, mimetype='application/json')

    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501)