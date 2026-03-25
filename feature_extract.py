import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Load the cleaned data
df = pd.read_csv('data/final_cleaned_news.csv')

# Check for NaN values in 'cleaned_text'
print("🔍 Checking for NaN values in 'cleaned_text'...")
print(f"Number of NaN values: {df['cleaned_text'].isna().sum()}")

# 👇 CRITICAL FIX 1: Remove rows with NaN or empty text
df = df.dropna(subset=['cleaned_text'])           # Remove NaN
df = df[df['cleaned_text'].str.strip() != '']     # Remove empty strings

print(f"✅ After cleaning: {len(df)} rows remaining")

# Features (X) and labels (y)
X = df['cleaned_text']   # Now guaranteed to be strings
y = df['label']

# Convert text to TF-IDF vectors (returns sparse matrix)
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

print("✅ Feature extraction complete!")
print(f"Total samples: {X_vec.shape[0]}")
print(f"Training samples: {X_train.shape[0]}")   # ✅ Use .shape[0] for sparse matrices
print(f"Test samples: {X_test.shape[0]}")        # ✅ Use .shape[0] for sparse matrices

# Save vectorizer and split data
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump((X_train, X_test, y_train, y_test), 'models/split_data.pkl')

print("✅ Saved vectorizer and data splits.")