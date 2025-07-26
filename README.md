# RAG AI Chatbot using LangChain and Streamlit

This project is a **Retrieval-Augmented Generation (RAG)** based question-answering app that allows you to upload `.json` documents, index them using **HuggingFace embeddings and FAISS**, and then query them using a **locally hosted LLaMA model**.

---

## 📁 Project Structure

```
├── app.py               # Streamlit frontend for uploading and querying documents
├── ingest.py            # Script to split, embed, and index documents using FAISS
├── rag_chain.py         # Constructs the QA chain using FAISS, HuggingFace embeddings, and LLaMA
├── constants.py         # Configuration constants used across the app
├── data/                # Folder for uploaded JSON files
├── faiss_index/         # FAISS index (auto-created after indexing)
└── README.md            # Project documentation
```

---

## ⚙️ Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

- `streamlit`
- `langchain`
- `faiss-cpu`
- `transformers`
- `llama-cpp-python` (or another LLM backend)
- `huggingface_hub`

---

## 📝 Setup Instructions

1. **Run the Streamlit App**

```bash
streamlit run app.py
```

2. **Index Documents**

- Upload one or more `.json` files through the UI
- Click **"Index Files"** to embed and store them using FAISS

3. **Ask Questions**

Once indexing is complete, enter any question related to the uploaded documents.

---

## ⚡️ How It Works

### `ingest.py`

- Loads JSON files
- Splits content into manageable text chunks
- Embeds using HuggingFace models
- Indexes using FAISS and saves to disk

### `rag_chain.py`

- Loads the FAISS index and embedding model
- Sets up a local LLaMA model
- Builds a `RetrievalQA` chain for answering questions

### `app.py`

- Streamlit interface for file upload and interaction
- Invokes `ingest.py` and `rag_chain.py` functionality dynamically
- Displays results directly in the browser

---

## 🔐 Notes

- For large local models, ensure you have enough RAM (ideally 8–16 GB).
- If using GPU acceleration, adjust `LlamaCpp` settings accordingly.
- Always validate the content of `.json` files before uploading.

---
