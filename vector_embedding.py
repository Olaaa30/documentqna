from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DOCUMENT_LIMIT = 20

def vector_embedding():
    """
    This function creates vector embeddings for documents in a specified directory.
    The documents are split into chunks for more efficient processing.
    The resulting vectors are stored in the Streamlit session state.
    """
    if "vectors" not in st.session_state:
        # Create the embeddings and save to a session
        st.session_state.embeddings = OpenAIEmbeddings()

        st.session_state.loader = PyPDFDirectoryLoader("./documents")

        # Error handling for document loading
        try:
            st.session_state.docs = st.session_state.loader.load()
        except FileNotFoundError:
            st.error("Document directory not found.")
            return

        # Split the documents into paragraphs
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:DOCUMENT_LIMIT])

        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)