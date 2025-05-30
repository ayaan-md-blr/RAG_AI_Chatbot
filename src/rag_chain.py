# rag_chain.py
import os
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
import constants

def create_qa_chain():
    """
    Creates a Retrieval-Augmented Generation (RAG) question-answering chain
    using a local LLaMA model, FAISS vector store, and HuggingFace embeddings.

    Returns:
        RetrievalQA: A LangChain QA chain object capable of answering questions
                     using embedded documents retrieved via FAISS.
    """
    # Local model
    llm = LlamaCpp(
        model_path=constants.LLM_MODEL_PATH,
        temperature=0.1,
        max_tokens=512,
        top_p=0.95,
        n_ctx=2048,
        n_threads=4,  # Tune this based on CPU
        verbose=False
    )

    # Initialize the HuggingFace embedding model
    embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDING_MODEL_NAME)

    # Load the FAISS index from local disk and link it to the embedding model
    db = FAISS.load_local(constants.FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)

    # Convert the FAISS vector store into a retriever interface
    retriever = db.as_retriever()

    # Create the retrieval-based QA chain using the LLM and retriever
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return chain
