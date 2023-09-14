import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables from a .env file
load_dotenv()

# Configure the Streamlit app
st.set_page_config(
    page_title="Educate Kids with Similar Words", page_icon="👩‍🎓"
)
st.header("Hey, Type a Word, and I'll suggest similar words to learn..")

# Initialize OpenAIEmbeddings and CSVLoader
embedding = OpenAIEmbeddings()
loader = CSVLoader(file_path="Data.csv", csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['Words']
})

# Load data from the CSV file
data = loader.load()

# Initialize FAISS vector store
db = FAISS.from_documents(data, embedding)

# Function to get user input


def get_text():
    input_text = st.text_input("You:")
    return input_text


# Get user input and trigger word similarity search
user_input = get_text()
submit = st.button("Find Similar Words")

if submit:
    docs = db.similarity_search(user_input)
    st.subheader("Top Matched Words:")
    st.text(docs[0].page_content)
    st.text(docs[1].page_content)
