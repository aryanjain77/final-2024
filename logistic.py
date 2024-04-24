import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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

def logistic_regression(input1):
    # Load the dataset
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

    # User input
    user_text = input1
    user_text_processed = preprocess_text(user_text)
    user_text_vectorized = vectorizer.transform([user_text_processed])

    # Prediction
    prediction = model.predict(user_text_vectorized)[0]
    if(prediction == 0) :
        label = "real"
    else :
        label = "fake"

    # Output result
    return label













# import csv
# import pandas as pd
# import re
# import sklearn
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder as le
# import nltk
# import joblib

# loaded_model = joblib.load('./model_saved')

# def stemming(content):
#      port_stem = PorterStemmer()
#      stemmed_content = re.sub("[^a-zA-Z]", " ", content)
#      stemmed_content = stemmed_content.lower()
#      stemmed_content = stemmed_content.split()
#      stemmed_content = [
#          port_stem.stem(word)
#          for word in stemmed_content
#          if not word in stopwords.words("english")
#      ]
#      stemmed_content = " ".join(stemmed_content)
#      return stemmed_content



# def handleInput(input1, input2, input3):

# # CSV file path
#     csv_file_path = 'train.csv'

# # Read existing data from CSV
#     existing_data = pd.read_csv(csv_file_path)

#     # Get the last row
#     last_row = existing_data.iloc[-1]

#     # Access the "ID" column of the last row
#     last_row_id = last_row['id']

#     new_row_id = last_row_id + 1

#     data = [{ "id" : new_row_id, "title" : input1, "author" : input2, "text" : input3, "labels" : None }]

# # Append new data
#     new_data = pd.DataFrame(data)
#     updated_data = pd.concat([existing_data, new_data], ignore_index=True)

# # Write updated data to CSV
#     updated_data.to_csv(csv_file_path, index=False)

#     print("Data appended successfully.")


#     user_dataset = pd.read_csv("train.csv")

#     # user_dataset["id"] = le.fit_transform(user_dataset["id"].astype(str))
#     # user_dataset["title"] = le.fit_transform(user_dataset["title"].astype(str))
#     # user_dataset["author"] = le.fit_transform(user_dataset["author"].astype(str))
#     # user_dataset["text"] = le.fit_transform(user_dataset["text"].astype(str))

#     user_dataset['id'] = user_dataset['id'].apply(str) 
    

#     user_dataset['title'] = user_dataset['title'].apply(str) 


#     user_dataset['author'] = user_dataset['author'].apply(str) 


#     user_dataset['text'] = user_dataset['text'].apply(str)
#     user_dataset['labels'] = user_dataset['labels'].apply(str) 


#     user_dataset = user_dataset.fillna("")

#     # merging the author name and user title
#     user_dataset["content"] = user_dataset["author"] + " " + user_dataset["title"]

#     # separating the data & label
#     X = user_dataset.drop(columns="labels", axis=1)
#     Y = user_dataset["labels"]


#     user_dataset["content"] = user_dataset["content"].apply(stemming)

#     # separating the data and label
#     X = user_dataset["content"].values
#     Y = user_dataset["labels"].values

#     # converting the textual data to numerical data
#     vectorizer = TfidfVectorizer()
#     vectorizer.fit(X)

#     X = vectorizer.transform(X)

#     X_train, X_test, Y_train, Y_test = train_test_split(
#         X, Y, test_size=0.2, stratify=Y, random_state=2
#     )

#     model = LogisticRegression()

#     model.fit(X_train, Y_train)

#     # accuracy score on the training data
#     X_train_prediction = model.predict(X_train)
#     training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

#     # accuracy score on the test data
#     X_test_prediction = model.predict(X_test)
#     test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

#     X_new = X[new_row_id]
#     prediction = loaded_model.predict(X_new)
#     if (prediction[0] == 0) :
#         res = "the user is real"
#     else :
#         res = "the user is fake"
#     return res
