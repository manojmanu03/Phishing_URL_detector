from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import re
from urllib.parse import urlparse
import tldextract
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load the model and feature names
try:
    model = joblib.load('models/phishing_model.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    print("Model and features loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please run train_model.py first to create the model!")
    model = None
    feature_names = None

def extract_features_from_url(url):
    """Extract features from URL matching the training data format"""
    features = {}
    
    try:
        # Initialize all features to 0
        if feature_names is None:
            raise ValueError("Feature names not loaded. Please run train_model.py first!")
            
        for feature in feature_names:
            features[feature] = 0
            
        # Parse URL
        parsed = urlparse(url)
        extract_res = tldextract.extract(url)
        domain = extract_res.domain + '.' + extract_res.suffix
        
        # Update features based on the phishing.csv columns
        # These should match your dataset's features
        
        # Example feature extraction (modify according to your actual features):
        if 'UsingIP' in features:
            features['UsingIP'] = 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', domain) else -1
            
        if 'PrefixSuffix-' in features:
            features['PrefixSuffix-'] = -1 if '-' in domain else 1
            
        if 'SubDomains' in features:
            subdomain_count = len(domain.split('.')) - 1
            features['SubDomains'] = 1 if subdomain_count <= 2 else -1
            
        if 'HTTPS' in features:
            features['HTTPS'] = 1 if parsed.scheme == 'https' else -1
            
        if 'DomainLength' in features:
            features['DomainLength'] = 1 if len(domain) < 54 else -1
            
        if 'Favicon' in features:
            features['Favicon'] = 1  # Default value, would need more complex analysis
            
        if 'NonStdPort' in features:
            port = parsed.port
            features['NonStdPort'] = -1 if port and port not in [80, 443] else 1
            
        if 'HTTPSDomainURL' in features:
            features['HTTPSDomainURL'] = 1 if parsed.scheme == 'https' else -1
            
        # Create a pandas DataFrame with feature names
        feature_df = pd.DataFrame([features], columns=feature_names)
        
        return feature_df, None
        
    except Exception as e:
        return None, str(e)

def is_url(url):
    """Check if string is a valid URL"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        try:
            if model is None:
                return jsonify({
                    'error': 'Model not loaded. Please run train_model.py first!'
                })
                
            url = request.form.get('url', '').strip()
            
            if not url:
                return jsonify({'error': 'Please enter a URL'})
            
            if not is_url(url):
                return jsonify({'error': 'Invalid URL format'})
            
            # Extract features
            features, error = extract_features_from_url(url)
            if error:
                return jsonify({'error': f'Error analyzing URL: {error}'})
            
            # Make prediction
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            
            # Convert numpy types to Python native types
            prediction = int(prediction)  # Convert np.int64 to int
            probabilities = [float(p) for p in probabilities]  # Convert np.float64 to float
            
            # Calculate confidence percentage
            if prediction == 1:  # Safe
                confidence = probabilities[1] * 100
            else:  # Unsafe
                confidence = probabilities[0] * 100
            
            return jsonify({
                'url': url,
                'confidence': f'{confidence:.2f}%',
                'is_phishing': prediction != 1  # Reverse the logic to match the UI expectations
            })
            
        except Exception as e:
            return jsonify({'error': f'Analysis error: {str(e)}'})
    
    return jsonify({'error': 'Invalid request method'})

if __name__ == '__main__':
    if model is None:
        print("WARNING: Model not loaded! Please run train_model.py first.")
    app.run(debug=True) 