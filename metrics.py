import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def preprocess_text(text):
    stop_words = stopwords.words("english")
    stemmer = PorterStemmer()
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = "".join([char for char in text if char.isalnum() or char.isspace()])
    # Remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])
    # Stemming
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text

data = pd.read_csv("train.csv")

data['title'] = data['title'].apply(str) 

data['text'] = data['text'].apply(str)

data.dropna(inplace=True,axis=0)

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

auc = roc_auc_score(y_test, model.predict_proba(X_test_vectorized)[:, 1])
accuracy = accuracy_score(y_test, model.predict(X_test_vectorized))
f1 = f1_score(y_test, model.predict(X_test_vectorized))

print("Accuracy:",accuracy)
print("F1-score:",f1)
print("Auc:",auc)