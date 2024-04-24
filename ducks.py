import numpy as np
import pandas as pd
import re
import sklearn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.stem.porter import PorterStemmer
import csv  

def stemming(content):
    
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmer=PorterStemmer()
    stemmer.stem(stemmed_content)
    # stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

def result(ui):
    nltk.download('stopwords')

    news_dataset = pd.read_csv('test1.csv')

    news_dataset = news_dataset.fillna('')

    # merging the author name and news title
    news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']

    # separating the data & label
    X = news_dataset.drop(columns='labels', axis=1)
    Y = news_dataset['labels']

    port_stem = PorterStemmer()

    # news_dataset['content'] = news_dataset['content'].apply(stemming)

    #separating the data and label
    X = news_dataset['title'].values
    Y = news_dataset['labels'].values

    # converting the textual data to numerical data
    vectorizer = TfidfVectorizer()
    vectorizer.fit(X)

    X = vectorizer.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)

    model = LogisticRegression()

    model.fit(X_train, Y_train)

    # accuracy score on the training data
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

    # accuracy score on the test data
    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

    X_new = X_test[0]

    prediction = model.predict(X_new)
    print(prediction)

    if (prediction[ui]==0):
      res = 'The news is Real'
    else:
      res = 'The news is fake'
    return res




import streamlit as st
from streamlit_lottie import st_lottie 
import json
  
path = "2.json"
with open(path,"r") as file: 
    url = json.load(file) 
  
  
  
st.title("machine learning is future shield to fake news") 
  
st_lottie(url, 
    reverse=True, 
    height=700, 
    width=700, 
    speed=1, 
    loop=True, 
    quality='high', 
    key='Car'
)
# st.set_page_config(page_title="fake news detector",layout="wide")
st.subheader("welcome to the revolution")
st.title("FAKE NEWS DETECTOR")
st.write("rru is smart")
lottie_coding=""
with st.container():
    st.write("-----")
    left_column,right_column=st.columns(2)
    with left_column:
        st.header("EXPLORE LEGITIMACY OF NEWS ")
        st.write("##")
        st.write("enter news to analyse")
        taker=st.text_input("type input")
        if st.button("submit", type="primary"):
                st.write(result(taker))
        
    with right_column:
        st.write("##")
        path3 = "1.json"
        with open(path3,"r") as file: 
           url = json.load(file) 
           st_lottie(url, 
           reverse=True, 
            height=700, 
            width=700, 
            speed=1, 
            loop=True, 
            quality='high', 
            key='Car3'
        )
       
       
        
path2 = "4.json"
with open(path2,"r") as file: 
    url = json.load(file) 
  
  
  
st.title("machine learning is future shield to fake news") 
  
st_lottie(url, 
    reverse=True, 
    height=700, 
    width=700, 
    speed=1, 
    loop=True, 
    quality='high', 
    key='Car2'
)