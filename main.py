# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv('./stress_dataset.csv')


# Convert categorical/textual data into numerical values using LabelEncoder
le = {}
for column in df.columns[:-1]:  # Exclude the target column
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    le[column] = encoder

# Splitting the dataset into training and testing sets (80-20 split)
X = df.drop("Stress Level", axis=1)
y = df["Stress Level"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Gradient Boosting classifier
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train, y_train)

# Predictions on the test set
y_pred_gb = gb_classifier.predict(X_test)

# Evaluate the model's performance
accuracy_gb = accuracy_score(y_test, y_pred_gb)
class_report_gb = classification_report(y_test, y_pred_gb)

# # Print results
# print(f"Accuracy: {accuracy_gb}")
# print("Classification Report:")
# print(class_report_gb)

joblib.dump(gb_classifier, 'gb_model.pkl')

joblib.dump(le, 'label_encoders.pkl')
