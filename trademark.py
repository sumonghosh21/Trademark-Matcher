

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import string
import base64

nltk.download('stopwords')


def preprocess_text(text, remove_punct, remove_stop):
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    if remove_punct:
        text = ''.join([char for char in text if char not in string.punctuation])
    if remove_stop:
        text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


def find_similar_trademarks(df, trademark, top_n, remove_punct, remove_stop, max_df, min_df):
    try:
        # Ensure unique trademarks
        df = df.drop_duplicates(subset=['trademark'])
        df['cleaned_trademark'] = df['trademark'].apply(lambda x: preprocess_text(x, remove_punct, remove_stop))
        vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df)
        tfidf_matrix = vectorizer.fit_transform(df['cleaned_trademark'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        cosine_sim_df = pd.DataFrame(cosine_sim, index=df['trademark'], columns=df['trademark'])

        # Check if the selected trademark is in the DataFrame
        if trademark not in cosine_sim_df.index:
            st.error(f"The selected trademark '{trademark}' is not present in the DataFrame.")
            return pd.Series()

        similar_trademarks = cosine_sim_df[trademark].sort_values(ascending=False)[1:top_n + 1]
        return similar_trademarks
    except Exception as e:
        st.error(f"An error occurred while finding similar trademarks: {e}")
        st.write(f"Exception details: {e}")
        return pd.Series()


def create_download_link(df):
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="similar_trademarks.csv">Download CSV file</a>'
        return href
    except Exception as e:
        st.error(f"An error occurred while creating the download link: {e}")
        return ""


st.set_page_config(page_title='Trademark Similarity Finder', layout='wide')

st.markdown("""
    <style>
    .reportview-container {
        background: #f8f9fa;
        padding: 20px;
    }
    .sidebar .sidebar-content {
        background: #343a40;
        color: white;
    }
    h1, h2, h3 {
        color: #343a40;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Trademark Similarity Finder')
st.write(
    'Upload a CSV or Excel file containing trademarks, and find similar trademarks using advanced text processing and similarity measures. Additional details like prior user, class number, and preparator are also considered.')

uploaded_file = st.file_uploader('Upload a CSV or Excel file', type=['csv', 'xlsx'])


def validate_file(file, file_type):
    try:
        if file_type == 'csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()

        # Check for 'trademark' column
        if 'trademark' not in df.columns:
            return "The uploaded file is missing the 'trademark' column."

        st.write("Columns in DataFrame after normalization:", df.columns)
        return None
    except pd.errors.ParserError:
        return "Error parsing the file. Please ensure it's a valid CSV or Excel file."
    except pd.errors.EmptyDataError:
        return "The uploaded file is empty."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


if uploaded_file is not None:
    file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'xlsx'
    validation_error = validate_file(uploaded_file, file_type)
    if validation_error:
        st.error(validation_error)
    else:
        try:
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Normalize column names
            df.columns = df.columns.str.strip().str.lower()

            # Ensure unique trademarks
            df = df.drop_duplicates(subset=['trademark'])

            st.write('## Data Preview')
            st.write(df.head())  # Print the first few rows for inspection
            st.write("Columns in DataFrame:", df.columns)

            # Check if 'trademark' column exists
            if 'trademark' not in df.columns:
                st.error("The DataFrame does not contain the 'trademark' column.")
            else:
                # Sidebar options
                st.sidebar.header('Options')
                remove_punct = st.sidebar.checkbox('Remove Punctuation', value=True)
                remove_stop = st.sidebar.checkbox('Remove Stopwords', value=True)
                show_preprocessed = st.sidebar.checkbox('Show Preprocessed Data', value=False)
                max_df = st.sidebar.slider('Max Document Frequency (max_df)', 0.0, 1.0, 1.0)
                min_df = st.sidebar.slider('Min Document Frequency (min_df)', 0, 10, 1)

                trademark = st.selectbox('Select a Trademark', df['trademark'].tolist())
                top_n = st.slider('Number of Similar Trademarks to Find', 1, 10, 3)

                if st.button('Find Similar Trademarks'):
                    with st.spinner('Computing similarities...'):
                        similar_trademarks = find_similar_trademarks(df, trademark, top_n, remove_punct, remove_stop,
                                                                     max_df, min_df)
                    if not similar_trademarks.empty:
                        st.write('## Similar Trademarks')
                        similar_df = pd.DataFrame({
                            'Trademark': similar_trademarks.index,
                            'Similarity': similar_trademarks.values
                        })
                        # Merge with original data to get additional details if present
                        if 'prior_user' in df.columns and 'class_number' in df.columns and 'preparator' in df.columns:
                            similar_df = similar_df.merge(df, left_on='Trademark', right_on='trademark')
                            st.write(
                                similar_df[['Trademark', 'Similarity', 'prior_user', 'class_number', 'preparator']])
                        else:
                            st.write(similar_df[['Trademark', 'Similarity']])

                        st.markdown(create_download_link(similar_df), unsafe_allow_html=True)

                    if show_preprocessed:
                        st.write('## Preprocessed Data')
                        st.write(df[['trademark', 'cleaned_trademark'] +
                                    [col for col in ['prior_user', 'class_number', 'preparator'] if col in df.columns]])
        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty.")
        except pd.errors.ParserError:
            st.error("The file could not be parsed. Please ensure it's properly formatted.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            st.write(f"Exception details: {e}")














