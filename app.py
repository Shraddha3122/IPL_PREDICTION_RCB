# Import libraries
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# ================================
# ✅ LOAD MODELS AND ENCODERS
# ================================
try:
    win_model = pickle.load(open('models/best_win_model.pkl', 'rb'))
    score_model = pickle.load(open('models/best_score_model.pkl', 'rb'))
    encoder = pickle.load(open('models/encoders.pkl', 'rb'))
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    print("✅ Models and encoders loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models or encoders: {e}")


# ================================
# 🔥 HANDLE UNSEEN LABELS SAFELY
# ================================
def safe_encode(label, encoder, default_value=-1):
    """Safely encode labels, handle unseen labels."""
    try:
        return encoder.transform([label])[0]
    except ValueError:
        print(f"⚠️ Warning: Unseen label '{label}' encountered. Assigning default value.")
        return default_value


# ================================
# 📊 PREDICT FIRST INNINGS SCORE
# ================================
@app.route('/predict/score', methods=['POST'])
def predict_score():
    try:
        # Get JSON data from request
        data = request.json

        # Validate input data
        required_keys = ['venue', 'team1', 'team2']
        if not all(key in data for key in required_keys):
            return jsonify({'error': f'Missing required parameters: {required_keys}'}), 400

        # Encode input data safely
        venue = safe_encode(data['venue'], encoder)
        team1 = safe_encode(data['team1'], encoder)
        team2 = safe_encode(data['team2'], encoder)

        # Reject request if any label is unknown
        if -1 in [venue, team1, team2]:
            return jsonify({'error': 'Unseen label encountered. Please verify input values.'}), 400

        # Prepare features for prediction
        features = np.array([[venue, team1, team2]])
        features_scaled = scaler.transform(features)

        # Predict score
        prediction = score_model.predict(features_scaled)[0]

        return jsonify({'predicted_score': int(prediction)})

    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500


# ================================
# 🎯 PREDICT MATCH WIN PROBABILITY
# ================================
@app.route('/predict/win', methods=['POST'])
def predict_win():
    try:
        # Get JSON data from request
        data = request.json

        # Validate input data
        required_keys = ['venue', 'toss_decision', 'team1', 'team2']
        if not all(key in data for key in required_keys):
            return jsonify({'error': f'Missing required parameters: {required_keys}'}), 400

        # Encode input data safely
        venue = safe_encode(data['venue'], encoder)
        toss_decision = safe_encode(data['toss_decision'], encoder)
        team1 = safe_encode(data['team1'], encoder)
        team2 = safe_encode(data['team2'], encoder)

        # Reject request if any label is unknown
        if -1 in [venue, toss_decision, team1, team2]:
            return jsonify({'error': 'Unseen label encountered. Please verify input values.'}), 400

        # Prepare features for prediction
        features = np.array([[venue, toss_decision, team1, team2]])

        # Predict win probability
        win_prob = win_model.predict_proba(features)[:, 1][0]

        return jsonify({'win_probability': round(win_prob, 4)})

    except Exception as e:
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500


# ================================
# 🏏 HEALTH CHECK ENDPOINT
# ================================
@app.route('/health', methods=['GET'])
def health_check():
    """Check API health status."""
    return jsonify({'status': 'API is running successfully'}), 200


# ================================
# 🏠 HOME ROUTE
# ================================
@app.route('/')
def home():
    """Home route."""
    return "🏏 IPL Prediction API is running! Use /predict/win or /predict/score."


# ================================
# 🎯 RUN FLASK APPLICATION
# ================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
