from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('models/fakenews_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

# TEST ROUTE 1 - Visit in browser
@app.route('/')
def home():
    return "🚀 Fake News API is running! Use POST /predict"

# TEST ROUTE 2 - Visit in browser  
@app.route('/test')
def test():
    return {"status": "OK", "models_loaded": True}

# YOUR MAIN API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    news_text = data['text']
    vec_text = vectorizer.transform([news_text])
    pred = model.predict(vec_text)[0]
    prob = model.predict_proba(vec_text)[0]
    
    return jsonify({
        'prediction': 'FAKE' if pred == 1 else 'REAL',
        'confidence': float(max(prob)),
        'confidence_fake': float(prob[1]),
        'confidence_real': float(prob[0])
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
