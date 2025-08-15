import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DataFrameLoader
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

DB_PATH = "vectorstores/db/"

def load_documents():
    try:
        print("📄 Loading patient CSV data...")
        patient_loader = DataFrameLoader(pd.read_csv('data/patient_data.csv'), page_content_column="history")
        patient_documents = patient_loader.load()
        for doc in patient_documents:
            doc.metadata['source'] = 'patient_data.csv'
        print(f"✅ Loaded {len(patient_documents)} patient documents.")
    except Exception as e:
        print(f"❌ Error loading patient_data.csv: {e}")
        return []

    pdf_documents = []
    try:
        print("📄 Loading PDF files from /data...")
        for filename in os.listdir('data'):
            if filename.endswith('.pdf'):
                print(f"   → Loading {filename}")
                loader = PyPDFLoader(os.path.join('data', filename))
                pdf_documents.extend(loader.load())
        print(f"✅ Loaded {len(pdf_documents)} PDF documents.")
    except Exception as e:
        print(f"❌ Error loading PDFs: {e}")

    return patient_documents + pdf_documents

def split_documents(documents):
    try:
        print("✂️ Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"❌ Error splitting documents: {e}")
        return []

def get_embedding_model():
    try:
        print("🧠 Loading embedding model...")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return None

def main():
    print("🚀 Starting data ingestion process...")
    
    documents = load_documents()
    if not documents:
        print("⚠️ No documents loaded. Exiting.")
        return
    
    chunks = split_documents(documents)
    if not chunks:
        print("⚠️ No chunks to embed. Exiting.")
        return

    embeddings = get_embedding_model()
    if embeddings is None:
        print("⚠️ Embedding model not loaded. Exiting.")
        return
    
    try:
        print(f"💾 Creating vector store at {DB_PATH}...")
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
        print("✅ Data ingestion complete!")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")

if __name__ == "__main__":
    main()
