import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

def load_data():
    return pd.read_csv('career_decision_data.csv')

def train_model():
    # Load the data
    data = load_data()
    
    # Separate features and target
    X = data.drop(columns=['status'])  # Features
    y = data['status']  # Target (Master's or Job)
    
    # Convert target to numeric
    y = (y == 'accept').astype(int)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Job', "Master's"]))
    
    # Save the trained model, scaler, and features
    with open('career_decision_model.pkl', 'wb') as f:
        pickle.dump((model, scaler, X.columns), f)

def load_model():
    with open('career_decision_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    if len(model_data) == 2:
        # Old format
        model, feature_names = model_data
        scaler = None
    else:
        # New format
        model, scaler, feature_names = model_data
    
    return model, scaler, feature_names

