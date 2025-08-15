import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DataFrameLoader
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

DB_PATH = "vectorstores/db/"

def load_documents():
    patient_loader = DataFrameLoader(
        pd.read_csv('data/patient_data.csv'),
        page_content_column="history"
    )
    patient_documents = patient_loader.load()
    for doc in patient_documents:
        doc.metadata['source'] = 'patient_data.csv'
    
    pdf_documents = []
    for filename in os.listdir('data'):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join('data', filename))
            pdf_documents.extend(loader.load())
            
    return patient_documents + pdf_documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

def main():
    print("Starting data ingestion process...")
    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = get_embedding_model()
    
    print(f"Creating vector store at {DB_PATH}...")
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_PATH
    )
    print("Data ingestion complete!")

if __name__ == "__main__":
    main()
