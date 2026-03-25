import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove @mentions and #hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Split into words, remove stopwords, stem
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Load cleaned data
import pandas as pd
df = pd.read_csv('data/cleaned_news.csv')

# Apply preprocessing
df['cleaned_text'] = df['combined_text'].apply(preprocess_text)

# Save again
df.to_csv('data/final_cleaned_news.csv', index=False)

print("✅ Text preprocessing complete!")
print(df[['combined_text', 'cleaned_text']].head())