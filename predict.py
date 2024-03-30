from flask import Flask, jsonify
import joblib
import warnings

warnings.filterwarnings("ignore", message="Invalid feature names.*")

app = Flask(__name__)

# Load the trained model - trained_model.pkl
model = joblib.load('trained_model.pkl')


@app.route('/predict/<int:input_value>', methods=['GET'])
def predict(input_value):
    prediction = model.predict([[input_value]])
    response = {'prediction': prediction[0]}
    print(response)
    return jsonify(response)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
