from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")


def vector_embedding():

    if "vectors" not in st.session_state:

        # Create the embeddings and save to a session
        st.session_state.embeddings = OpenAIEmbeddings()

        st.session_state.loader = PyPDFDirectoryLoader("./documents")

        st.session_state.docs = st.session_state.loader.load()

        # Split the documents into paragraphs
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
