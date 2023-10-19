from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# Load the trained Gradient Boosting model and LabelEncoders
gb_classifier = joblib.load('./gb_model.pkl')
le = joblib.load('./label_encoders.pkl')

app = Flask(__name__)

# Define recommendations for each stress level
recommendations_dict = {
    "Normal": [
        "Continue maintaining a healthy work-life balance.",
        "Regular breaks during work can help maintain this state.",
        "Consider engaging in relaxation exercises or hobbies."
    ],
    "Mild": [
        "Take short breaks during work hours to relax.",
        "Engage in physical activity; even a short walk can help.",
        "Consider discussing any work concerns with your supervisor."
    ],
    "Moderate": [
        "Consider speaking to a professional or counselor about stress.",
        "Engage in regular physical and relaxation activities.",
        "It may be beneficial to consider vacation or time off."
    ],
    "Severe": [
        "It's crucial to seek professional help or counseling.",
        "Discuss your workload and concerns with your supervisor.",
        "Consider taking time off for mental well-being."
    ],
    "Extremely Severe": [
        "Immediate professional intervention is recommended.",
        "Consider discussing with HR about potential solutions or accommodations.",
        "Prioritize mental health; taking a break or leave might be necessary."
    ]
}

@app.route('/')
def index():
    print('Request for index page received')
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        df = pd.read_csv('./stress_dataset.csv')
        
        # Preprocess the data using the saved LabelEncoders
        for column, encoder in le.items():
            data[column] = encoder.transform([data[column]])[0]
        
        # Convert processed data to a list of values for prediction
        input_data = [data[column] for column in df.columns[:-1]]

        # Make a prediction using the Gradient Boosting model
        prediction = gb_classifier.predict([input_data])[0]

        # Get recommendations based on prediction
        prediction_recommendations = recommendations_dict[prediction]

        # Create a JSON response with the prediction and recommendations
        response = {
            'prediction': prediction,
            'recommendations': prediction_recommendations
        }
        return jsonify(response), 200
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run()
