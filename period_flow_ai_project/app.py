from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'flow_model.pkl')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("flow_model.pkl not found. Run 'python model.py' to create it.")

with open(MODEL_PATH,'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        features = np.array([
            int(data.get('cycle_day',0)),
            int(data.get('cramps',0)),
            int(data.get('mood',2)),
            int(data.get('bloating',0)),
            int(data.get('prev_flow',2))
        ]).reshape(1, -1)
        prediction = model.predict(features)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features).tolist()
        return jsonify({'flow_intensity': prediction, 'probabilities': proba})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
