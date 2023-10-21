import pandas as pd
import streamlit as st
import gensim
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.express as px
from typing import List, Tuple

st.set_page_config(
        page_title="Clinical Data Trials Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded"
    )


# Load data
@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv('clean.csv', encoding='latin1')
    return data

# Define functions for data filtering and text preprocessing
def filter_data(data: pd.DataFrame, column: str, values: List[str]) -> pd.DataFrame:
    return data[data[column].isin(values)] if values else data

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

data = load_data()
data['Tokenized Text'] = data['Brief Summary'].apply(preprocess_text)

# Train Word2Vec model
wordvec = gensim.models.Word2Vec(data['Tokenized Text'], vector_size=300)




@st.cache_data
def calculate_kpis(data: pd.DataFrame) -> List[float]:
    total_enrollment = data['Enrollment'].sum()
    total_trials = len(data)
    avg_enrollment = total_enrollment / total_trials
    return [total_enrollment, total_trials, avg_enrollment]

# Define the main function
def main():
  
    st.title("Clinical Data Trials")
    st.write("This is a machine learning model that helps researchers/students to get studies of clinical trials based on the disease or symptoms")

    nav = st.sidebar.radio("Navigation", ['Home', 'Dashboard', 'Classifier'])

    if nav == "Home":
        if st.button("List of Clinical Data Trials"):
            st.write(data[["Study Title", "Study URL", "Brief Summary"]].head(10))

    elif nav == "Classifier":
        st.subheader("What would you like to purchase?")
        product_name = st.text_input("Enter the product name:")
        # Preprocess the input
        product_tokens = preprocess_text(product_name)
    
        # Check if the list of tokens is empty
        if not product_tokens:
             st.write("Please enter a valid product name.")
        else:
            # Find similar words to the entered product name
            sim_words = wordvec.wv.most_similar(product_tokens, topn=5)

            # Filter and display the matched studies
            st.subheader("Matched Studies:")
            for word, cos in sim_words:
                matched_studies = data[data['Tokenized Text'].apply(lambda x: word in x)]
                for index, row in matched_studies.iterrows():
                    st.write("Study Title:", row['Study Title'])
                    st.write("Study URL:", row['Study URL'])
                    st.write("Study Status:", row['Study Status'])



    elif nav == "Dashboard":
        data = load_data()

        st.sidebar.header("Filters")    
        selected_countries = st.sidebar.multiselect("Select Countries", data['country'].unique())
        selected_categories = st.sidebar.multiselect("Select Category", data['category'].unique())

        filtered_data = data.copy()
        filtered_data = filter_data(filtered_data, 'category', selected_categories)
        filtered_data = filter_data(filtered_data, 'country', selected_countries)

        kpis = calculate_kpis(filtered_data)
        kpi_names = ["Total Enrollment", "Total Trials", "Average Enrollment per Trial"]

        st.header("KPI Metrics")
        for kpi_name, kpi_value in zip(kpi_names, kpis):
            st.metric(label=kpi_name, value=kpi_value)

        combine_product_lines = st.checkbox("Combine Product Lines", value=True)

        if combine_product_lines:
            fig = px.area(filtered_data, x='Start Date', y='Enrollment', title="Enrollment Over Time", width=900, height=500)
        else:
            fig = px.area(filtered_data, x='Start Date', y='Enrollment', color='PRODUCTLINE', title="Enrollment Over Time", width=900, height=500)

        fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        fig.update_xaxes(rangemode='tozero', showgrid=False)
        fig.update_yaxes(rangemode='tozero', showgrid=True)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
     main()
