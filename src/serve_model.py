"""
Flask API of the SMS Spam detection model model.
"""
import joblib
import argparse
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd

from text_preprocessing import prepare, _extract_message_len, _text_process

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether an SMS is Spam.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: sms
            properties:
                sms:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: 'spam' or 'ham'."
    """
    input_data = request.get_json()
    sms = input_data.get('sms')
    processed_sms = prepare(sms)
    model = joblib.load('output/model.joblib')
    prediction = model.predict(processed_sms)[0]
    
    res = {
        "result": prediction,
        "classifier": "decision tree",
        "sms": sms
    }
    print(res)
    return jsonify(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMS Spam Detection Model Service')
    parser.add_argument('--port', type=int, default=8081, help='Port to run the service on (default: 8081)')
    args = parser.parse_args()
    
    port = args.port
    print(f"Starting model service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
