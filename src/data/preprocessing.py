# creating preprocessed_data.pkl with text preprocessing and train-test split
# 01.12 --> now using training_data_cleaned.csv instead of training_data.csv
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the dataset
df = pd.read_csv("training_data_cleaned.csv")  # Should contain 'passage' and 'label' columns

# Function to clean and normalize text
def preprocess_text(text):
    text = text.lower().strip()  # Lowercase & trim spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # Remove stopwords
    return text

# Apply text preprocessing
df["cleaned_passage"] = df["passage"].astype(str).apply(preprocess_text)

# Convert text to numerical features (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000)  # Use top 5000 words
X = vectorizer.fit_transform(df["cleaned_passage"])  # Transform text into vectors
y = df["label"]  # Target labels (0 = negative, 1 = positive)
sentences = df["passage"]  # Original sentences

# Split into Train (80%), Validation (10%), Test (10%)
X_train, X_temp, y_train, y_temp, train_sentences, temp_sentences = train_test_split(
    X, y, sentences, test_size=0.2, random_state=42
)

X_val, X_test, y_val, y_test, val_sentences, test_sentences = train_test_split(
    X_temp, y_temp, temp_sentences, test_size=0.5, random_state=42
)

# Save preprocessed data and sentences
pickle.dump(
    (X_train, y_train, X_val, y_val, X_test, y_test, test_sentences),
    open("preprocessed_data2.pkl", "wb")
)

pickle.dump(vectorizer, open("tfidf_vectorizer2.pkl", "wb"))

print("Preprocessing complete. Data and sentences saved as 'preprocessed_data2.pkl'.")
