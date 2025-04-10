from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('model/model.pkl')

# Descriptions for A1 to A10
questions = {
    1: "Difficulty in social interaction",
    2: "Tendency to avoid eye contact",
    3: "Preference for routine",
    4: "Sensitivity to sensory input",
    5: "Difficulty understanding emotions",
    6: "Engages in repetitive behaviors",
    7: "Unusual speech patterns",
    8: "Challenges in making friends",
    9: "Limited interests",
    10: "Resistance to change"
}

@app.route('/')
def home():
    # ✅ Pass the function here!
    return render_template('index.html', get_description=lambda i: questions[i])

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[key]) for key in request.form]
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    result = 'Positive for Autism' if prediction[0] == 1 else 'Negative for Autism'
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
