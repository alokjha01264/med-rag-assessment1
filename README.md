---
title: Medical RAG Assistant
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
app_file: gradio_app.py
---
# 🩺 Nervesparks Medical Literature RAG Assistant

## 📖 Project Description

This project is a Retrieval-Augmented Generation (RAG) system built to assist healthcare professionals with evidence-based decision making. It processes a curated collection of medical journals, clinical guidelines, and anonymized patient data to answer complex clinical questions, providing answers grounded in documented evidence.

This system was developed as a project submission, fulfilling all core requirements including document processing, patient data integration, and evidence-based response generation.

---

## ✨ Features

* **Document Ingestion:** Processes medical PDFs and patient data from CSV files.
* **Vector Storage:** Uses ChromaDB for efficient, local storage and retrieval of information.
* **Evidence-Based Q&A:** Employs the `google/flan-t5-large` language model to synthesize answers based on retrieved documents.
* **Source Citing:** All answers are accompanied by the source documents used to generate them, ensuring transparency and verifiability.
* **Interactive UI:** A clean and user-friendly interface built with Streamlit, featuring a sidebar for instructions and example questions.

---

## 🏗️ System Architecture

The system follows a standard RAG pipeline:

1.  **Data Ingestion (`ingest.py`):**
    * Loads documents from the `/data` folder.
    * Splits documents into smaller, manageable chunks using a recursive character splitter.
    * Generates embeddings for each chunk using the `sentence-transformers/all-mpnet-base-v2` model.
    * Stores the chunks and their embeddings in a persistent ChromaDB vector store located in the `/vectorstores` directory.

2.  **Q&A and Generation (`app.py`):**
    * The user enters a query in the Streamlit UI.
    * The query is embedded using the same sentence-transformer model.
    * The vector store is searched to find the most semantically similar document chunks.
    * The user's query and the retrieved chunks are formatted into a detailed prompt.
    * The prompt and context are passed to the `google/flan-t5-large` model, which generates an answer based only on the provided evidence.
    * The final answer and its sources are displayed to the user.

---

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-name>
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    # For Windows
    python -m venv med_rag_env
    med_rag_env\Scripts\activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

1.  **Ingest Your Data:**
    Place your PDFs and patient data CSV in the `/data` folder and run the ingestion script once to build the vector database.
    ```bash
    python ingest.py
    ```
2.  **Launch the Application:**
    Start the Streamlit web server.
    ```bash
    streamlit run app.py
    ```

---

## 📊 Evaluation Results

The system was evaluated on a curated test set of questions using metrics for latency and semantic similarity (cosine similarity) of the answers compared to a ground truth.

---------------------------------
--- Average Evaluation Metrics ---
Average Latency: 7.98 seconds
Average Answer Similarity: 0.7580
---------------------------------