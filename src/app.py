import streamlit as st
from rag_chain import create_qa_chain
import os
import constants

# Set Streamlit page configuration
st.set_page_config(page_title="RAG Demo", layout="centered")
st.title("🧠 Ask Your Docs")

# File uploader widget to accept multiple .json files
uploaded_files = st.file_uploader("Upload .json files", accept_multiple_files=True, type=[constants.JSON])

# Process uploaded files
if uploaded_files:
    os.makedirs("data", exist_ok=True)
    for file in uploaded_files:
        with open(os.path.join(constants.DATA, file.name), "wb") as f:
            f.write(file.read())
    st.success("Files uploaded successfully!")

    if st.button("Index Files"):

        # Import ingest_documents lazily to avoid unnecessary overhead
        from ingest import ingest_documents

        # Run document ingestion and indexing pipeline
        ingest_documents()
        st.success("Documents indexed!")

# Enable question-answering only if FAISS index exists
if os.path.exists(constants.FAISS_INDEX):
    query = st.text_input("Ask a question about your documents:")
    if query:
        # Create the QA chain pipeline (retriever + LLM)
        chain = create_qa_chain()

        # Execute the chain with the user's query
        result = chain.run(query)
        st.markdown(f"**Answer:** {result}")