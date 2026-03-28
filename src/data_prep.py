
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path):
	"""Load the email dataset from a CSV file."""
	return pd.read_csv(path)

def clean_text(text):
	import string
	from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
	text = text.lower()
	text = text.translate(str.maketrans('', '', string.punctuation))
	words = text.split()
	words = [w for w in words if w not in ENGLISH_STOP_WORDS]
	return ' '.join(words)

def preprocess_data(df, text_col='text'):
	"""Clean the text column and add a new column 'clean_text'."""
	df['clean_text'] = df[text_col].apply(clean_text)
	return df

def vectorize_text(df, text_col='clean_text'):
	"""Convert text to TF-IDF features."""
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(df[text_col])
	return X, vectorizer

def split_data(X, y, test_size=0.2, random_state=42):
	"""Split data into train and test sets."""
	return train_test_split(X, y, test_size=test_size, random_state=random_state)