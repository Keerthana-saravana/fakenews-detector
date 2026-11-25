import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Create models folder
os.makedirs('models', exist_ok=True)

print("Loading dataset...")
fake = pd.read_csv('dataset/dataset/politifact_fake.csv')
real = pd.read_csv('dataset/dataset/politifact_real.csv')

print("Columns found:", fake.columns.tolist())

# Use TITLE only (FakeNewsNet format - no 'text' column)
fake['text'] = fake['title'].fillna('')
real['text'] = real['title'].fillna('')

fake['label'] = 1
real['label'] = 0
df = pd.concat([fake, real], ignore_index=True)

print(f"Dataset loaded: {len(df)} samples")

# Preprocess + Train
print("Training model...")
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X = vectorizer.fit_transform(df['text'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
preds = model.predict(X_test)

# Results (SCREENSHOT THESE)
accuracy = accuracy_score(y_test, preds)
print(f"\n🎯 ACCURACY: {accuracy:.2%}")
print("\n📊 Classification Report:")
print(classification_report(y_test, preds))
print("\n📈 Confusion Matrix:")
print(confusion_matrix(y_test, preds))

# Save models
joblib.dump(model, 'models/fakenews_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("\n💾 Models saved!")

# Prediction function for Day 2 API
def predict_news(text):
    vec_text = vectorizer.transform([text])
    pred = model.predict(vec_text)[0]
    prob = model.predict_proba(vec_text)[0].max()
    return "FAKE" if pred == 1 else "REAL", prob

# Test 3 samples
tests = [
    "Trump arrested by FBI at Mar-a-Lago",
    "Weather forecast predicts rain tomorrow", 
    "Biden announces free money for voters"
]
for news in tests:
    result, conf = predict_news(news)
    print(f"\n🧪 '{news}' → {result} ({conf:.1%} confidence)")
