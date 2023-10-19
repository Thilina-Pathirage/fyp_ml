from flask import Flask, request, jsonify
import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the trained Gradient Boosting model and LabelEncoders
gb_classifier = joblib.load('./gb_model.pkl')
le = joblib.load('./label_encoders.pkl')

app = Flask(__name__)



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

        # Create a JSON response with the prediction
        response = {'prediction': prediction}
        return jsonify(response), 200
    except Exception as e:
        return str(e), 400

if __name__ == '__main__':
    app.run()
