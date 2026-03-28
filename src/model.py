
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_logistic_regression(X_train, y_train):
	"""Train a logistic regression model."""
	model = LogisticRegression(max_iter=1000)
	model.fit(X_train, y_train)
	return model

def evaluate_model(model, X_test, y_test):
	"""Evaluate the model and print metrics."""
	y_pred = model.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	print(f"Accuracy: {acc:.4f}")
	print(classification_report(y_test, y_pred))
	return acc

def save_model(model, path):
	"""Save the trained model to disk."""
	joblib.dump(model, path)

def load_model(path):
	"""Load a trained model from disk."""
	return joblib.load(path)