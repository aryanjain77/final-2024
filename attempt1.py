import streamlit as st
from streamlit_lottie import st_lottie
import json
from logistic import logistic_regression

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
lottie_coding=""
with st.container():
    st.write("-----")
    left_column,right_column=st.columns(2)
    with left_column:
        st.header("EXPLORE LEGITIMACY OF NEWS ")
        st.write("##")
        st.write("enter news to analyse")
        text=st.text_input("type your news here")
        st.write("##")
        if st.button("submit", type="primary"):
            st.write(f"the news is classified as: {logistic_regression(text)}")
        # st.write("##")
        # st.write("previous one")
        # taker=st.number_input("type input",value=0,step=1,format="%d")
        # if st.button("submit", type="primary"):
        #     st.write(predict(taker))


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
