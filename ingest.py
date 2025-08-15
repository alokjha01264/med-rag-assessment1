import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DataFrameLoader
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

DB_PATH = "vectorstores/db/"

def load_documents():
    all_documents = []

  
    try:
        csv_path = "data/patient_data.csv"
        if os.path.exists(csv_path):
            print(f"📄 Loading CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            if "history" not in df.columns:
                raise ValueError(f"'history' column missing in {csv_path}")
            patient_loader = DataFrameLoader(df, page_content_column="history")
            patient_documents = patient_loader.load()
            for doc in patient_documents:
                doc.metadata['source'] = 'patient_data.csv'
            print(f"✅ Loaded {len(patient_documents)} patient documents.")
            all_documents.extend(patient_documents)
        else:
            print(f"⚠️ CSV file not found: {csv_path}")
    except Exception as e:
        print(f"❌ Error loading patient_data.csv: {e}")

    try:
        print("📄 Checking for PDF files in /data...")
        pdf_documents = []
        for filename in os.listdir("data"):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join("data", filename)
                print(f"   → Loading PDF: {file_path}")
                loader = PyPDFLoader(file_path)
                pdf_documents.extend(loader.load())
        if pdf_documents:
            print(f"✅ Loaded {len(pdf_documents)} PDF documents.")
            all_documents.extend(pdf_documents)
        else:
            print("⚠️ No PDF files found in /data.")
    except Exception as e:
        print(f"❌ Error loading PDFs: {e}")

    return all_documents

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
        print("❌ No documents loaded. Add a CSV or PDF to /data and try again.")
        return
    
    chunks = split_documents(documents)
    if not chunks:
        print("❌ No chunks created. Check your document formats.")
        return

    embeddings = get_embedding_model()
    if embeddings is None:
        print("❌ Embedding model failed to load. Check your internet connection or model name.")
        return
    

    os.makedirs(DB_PATH, exist_ok=True)
    
    try:
        print(f"💾 Creating vector store at {DB_PATH}...")
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_PATH)
        print("✅ Data ingestion complete! Vector DB is ready.")
    except Exception as e:
        print(f"❌ Error creating vector store: {e}")

if __name__ == "__main__":
    main()
