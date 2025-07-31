# Phishing URL Detector

A modern web application that detects potential phishing URLs using machine learning.

## Features

- Real-time URL analysis
- Machine learning-based detection
- Modern, responsive UI
- Detailed risk assessment
- Multiple feature analysis including:
  - Domain analysis
  - URL structure analysis
  - SSL/TLS verification
  - Domain age checking
  - Suspicious TLD detection
  - And more...

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd phishing_detector
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training the Model

1. Prepare your dataset (if you have one) or use the sample data in `train_model.py`

2. Run the training script:
```bash
python train_model.py
```

This will create a trained model in the `models` directory.

## Running the Application

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Enter a URL in the input field
2. Click "Analyze URL"
3. View the detailed analysis results

## Technical Details

The application uses:
- Flask for the web framework
- Scikit-learn for machine learning
- Random Forest Classifier for prediction
- Feature extraction based on URL characteristics
- Bootstrap 5 for the UI
- AJAX for asynchronous requests

## Features Analyzed

The system analyzes various URL features including:
- URL length and structure
- Domain name characteristics
- SSL/TLS certificate
- Domain age
- Suspicious TLD detection
- Number of dots, hyphens, and special characters
- Protocol used (HTTP/HTTPS)
- Presence of IP addresses
- Subdomain analysis
- And more...

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 