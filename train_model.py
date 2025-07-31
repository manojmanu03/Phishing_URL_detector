import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import shutil

def train_model():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load and prepare the dataset
    print("Loading dataset...")
    
    # Copy the phishing.csv file from parent directory if it exists
    if os.path.exists('../phishing.csv'):
        shutil.copy('../phishing.csv', 'phishing.csv')
    
    # Load the dataset
    df = pd.read_csv('phishing.csv')
    
    # Prepare features and target
    print("Preparing features...")
    
    # All columns except the last one (assuming last column is the target)
    feature_columns = df.columns[:-1]
    X = df[feature_columns]
    y = df[df.columns[-1]]  # Last column is the target
    
    # Split the dataset
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, 
                                 max_depth=10,
                                 min_samples_split=5,
                                 min_samples_leaf=2,
                                 random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    print("\nModel Evaluation:")
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    print("\nSaving model...")
    model_path = 'models/phishing_model.joblib'
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}!")
    
    # Save feature names
    feature_names = list(feature_columns)
    joblib.dump(feature_names, 'models/feature_names.joblib')
    print("Feature names saved successfully!")
    
    return model, feature_names

if __name__ == "__main__":
    train_model() 