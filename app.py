import pandas as pd
import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import plotly.express as px
model = Word2Vec.load("wordvec.model")
data = pd.read_csv('clean.csv')
    
def preprocess_text(text):
    # Remove non-alphabetic characters, lowercase, and tokenize
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    #tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    return text


# Load data

nav = st.sidebar.radio("Menu", ['Home', 'Classifier'])

if nav == "Home":
    st.title("Clinical Data Trials")
    st.subheader("Overview")
    st.write("Clinical trials are a crucial component of the medical and healthcare research process.They are systematic investigations involving human participants designed to evaluate the safety, efficacy, and side effects of new medical treatments, interventions, drugs, or devices.")
    if st.button("List of Clinical Data Trials"):
        st.write(data[["Study Title", "Study URL", "Brief Summary"]].head(10))

    year_counts = data['start_year'].value_counts().sort_index()  
    years = year_counts.index
    counts = year_counts.values

    # Plotting the data using Plotly Express
    fig = px.line(x=years, y=counts, markers=True, line_shape="linear")
    fig.update_layout(
        title="Clinical Trials Over the Years",
        xaxis_title="Year",
        yaxis_title="Number of Trials",
        width=800,
        height=400
    )
    st.plotly_chart(fig)  
    
   
elif nav == "Classifier":
    st.write("This is machine learning model  that streamlines the search for clinical trials based on specific medical conditions or research interests.")
    st.subheader("What would you like to research about?")
    disease_name = st.text_input("Enter the disease/condition name:")

    if st.button("Find Related Studies"):
        # Find similar words to the entered disease name
        sim_words = model.wv.most_similar(disease_name.lower(), topn=5)

        # Filter and display the matched studies
        st.subheader("Related Studies:")
        for word, cos in sim_words:
            matched_studies = data[data['Tokenized Text'].apply(lambda x: word in x)]
            for index, row in matched_studies.iterrows():
                st.write("Study Title:", row['Study Title'])
                st.write("Study URL:", row['Study URL'])
                st.write("Study Status:", row['Study Status'])


    