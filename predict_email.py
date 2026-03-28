import joblib
import os

def clean_text(text):
    import string
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

# Load model
model = joblib.load(os.path.join('models', 'spam_classifier.pkl'))
# Load vectorizer
vectorizer = joblib.load(os.path.join('models', 'vectorizer.pkl'))

# Get email input from user
user_email = input("Enter the email text: ")
clean_email = clean_text(user_email)
X_new = vectorizer.transform([clean_email])

# Predict
prediction = model.predict(X_new)
print("Prediction:", prediction[0])  # Output will be 'spam' or 'ham'
