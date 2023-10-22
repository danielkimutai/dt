import pandas as pd
import streamlit as st
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
model = Word2Vec.load("wordvec.model")
data = pd.read_csv('clean.csv')
    
def preprocess_text(text):
    # Remove non-alphabetic characters, lowercase, and tokenize
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


# Load data

data['Tokenized Text'] = data['Brief Summary'].apply(preprocess_text)
nav = st.sidebar.radio("Navigation", ['Home', 'Classifier'])

if nav == "Home":
    st.title("Clinical Data Trials")
    st.write("This is a machine learning model that helps researchers/students to get studies of clinical trials based on the disease or symptoms")
    if st.button("List of Clinical Data Trials"):
        st.write(data[["Study Title", "Study URL", "Brief Summary"]].head(10))


elif nav == "Classifier":
    st.subheader("What would you like to research about?")
    product_name = st.text_input("Enter the disease/conditon name:")
    # Find similar words to the entered product name
    sim_words = model.wv.most_similar(product_name, topn=5)

        # Filter and display the matched studies
    st.subheader("Matched Studies:")
    for word, cos in sim_words:
        matched_studies = data[data['Tokenized Text'].apply(lambda x: word in x)]
        for index, row in matched_studies.iterrows():
                st.write("Study Title:", row['Study Title'])
                st.write("Study URL:", row['Study URL'])
                st.write("Study Status:", row['Study Status'])



    