from src.data_prep import load_data, preprocess_data, vectorize_text, split_data
from src.model import train_logistic_regression, evaluate_model, save_model
import os

# 1. Load data

# Try both possible locations for the dataset
raw_data_path = os.path.join('data', 'raw', 'email_spam.csv')
if not os.path.exists(raw_data_path):
	raw_data_path = os.path.join('data', 'email_spam.csv')
df = load_data(raw_data_path)


# 2. Preprocess data (use 'v2' for text column)
df = preprocess_data(df, text_col='v2')

# 3. Vectorize text
X, vectorizer = vectorize_text(df, text_col='clean_text')

# 4. Prepare labels (use 'v1' for label column)
y = df['v1']

# 5. Split data
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = train_logistic_regression(X_train, y_train)

# 7. Evaluate model
evaluate_model(model, X_test, y_test)

# 8. Save model
save_model(model, os.path.join('models', 'spam_classifier.pkl'))
