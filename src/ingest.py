# ingest.py
import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import constants

def ingest_documents(folder_path=constants.DATA):
    """
    Loads, splits, embeds, and indexes documents from the given folder path
    using FAISS and HuggingFace embeddings.
    """
    
    # Load text documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(constants.JSON_EXT):

            # Load the document using LangChain's TextLoader
            loader = TextLoader(os.path.join(folder_path, filename))
            docs = loader.load()

            # Split the document into smaller chunks
            chunks = text_splitter.split_documents(docs)
            documents.extend(chunks)

    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name=constants.EMBEDDING_MODEL_NAME)

    # Create and persist FAISS index
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(constants.FAISS_INDEX)
    print("✅ Documents indexed and saved to 'faiss_index'.")

if __name__ == "__main__":
    ingest_documents()
