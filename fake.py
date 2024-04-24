import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the dataset
data = pd.read_csv("train.csv")

# Define stopwords and stemmer
stop_words = stopwords.words("english")
stemmer = PorterStemmer()

data['title'] = data['title'].apply(str) 

data['text'] = data['text'].apply(str)

data.dropna(inplace=True,axis=0)

# Function for text preprocessing
def preprocess_text(text):
  # Lowercase
  text = text.lower()
  # Remove punctuation
  text = "".join([char for char in text if char.isalnum() or char.isspace()])
  # Remove stopwords
  text = " ".join([word for word in text.split() if word not in stop_words])
  # Stemming
  text = " ".join([stemmer.stem(word) for word in text.split()])
  return text

# Preprocess text columns
data["title"] = data["title"].apply(preprocess_text)
data["text"] = data["text"].apply(preprocess_text)

# Combine title and text for features
data["combined_text"] = data["title"] + " " + data["text"]

# Separate features and labels
X = data["combined_text"]
y = data["labels"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=2000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Logistic regression model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# User input
user_text = input("Enter news text to classify: ")
user_text_processed = preprocess_text(user_text)
user_text_vectorized = vectorizer.transform([user_text_processed])

# Prediction
prediction = model.predict(user_text_vectorized)[0]
label = "Real" if prediction == 0 else "Fake"

# Output result
print(f"The news is classified as: {label}")