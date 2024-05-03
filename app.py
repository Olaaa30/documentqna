import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from vector_embedding import vector_embedding
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


load_dotenv() 
#load environment variables for openai and groq
groq_api_key = os.environ.get("GROQ_API_KEY")


st.title("Chatgroq with Llama3 Demo")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>

    Questions: {input}
    """
    
)



first_prompt = st.text_input("Enter Your Question From The Document")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB is ready for use")

import time

if first_prompt:
    document_chain=create_stuff_documents_chain(llm, prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response=retrieval_chain.invoke({"input":first_prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")