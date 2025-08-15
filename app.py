import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch
import os

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

with st.sidebar:
    st.title("🩺 Medical RAG Assistant")
    st.markdown("""
    This app is a Retrieval-Augmented Generation (RAG) system designed to assist healthcare professionals. 
    It can answer questions based on a curated set of medical documents and synthetic patient data.
    """)
    
    st.markdown("---")
    
    st.subheader("💡 How to Use")
    st.info("""
    1.  Enter a clinical question in the text box.
    2.  For best results, ask clear, focused questions about a single topic.
    3.  The system will retrieve relevant information from its knowledge base and generate a synthesized, evidence-based answer.
    4.  You can review the source documents used for the answer in the expandable sections below the main response.
    """)

    st.markdown("---")

    st.subheader("Example Questions")
    example_questions = [
        "What are the 2017 AHA guideline recommendations for managing high blood pressure?",
        "What is the approximate heritability of Major Depressive Disorder?",
        "What is the primary therapeutic goal in the management of Rheumatoid Arthritis?"
    ]
    
    for question in example_questions:
        if st.button(question):
            st.session_state.question_input = question

    st.markdown("---")
    st.info("This is a proof-of-concept and not for clinical use.")

DB_PATH = "vectorstores/db/"
MODEL_ID = "google/flan-t5-large"

@st.cache_resource
def load_embedding_model():
    print("Loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def load_llm():
    print("Loading Language Model...")
    return HuggingFacePipeline.from_model_id(
        model_id=MODEL_ID,
        task="text2text-generation",
        model_kwargs={"temperature": 0.1, "max_length": 1024}
    )

@st.cache_resource
def load_rag_chain():
    embeddings = load_embedding_model()
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    llm = load_llm()

    template = """
    You are an expert medical AI assistant. Your task is to provide a detailed, evidence-based answer to the user's question based ONLY on the provided context.
    INSTRUCTIONS:
    - Read the following context carefully.
    - Synthesize the information from all provided sources into a single, coherent paragraph.
    - If the context does not contain enough information to answer the question, you MUST state that clearly. Do not make up information.
    - Cite the sources used in your answer by referencing the source name (e.g., `patient_data.csv` or the PDF name).
    CONTEXT:
    {context}
    QUESTION:
    {question}
    DETAILED, EVIDENCE-BASED ANSWER:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    return RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

st.header("Ask a Clinical Question")

if 'question_input' not in st.session_state:
    st.session_state.question_input = ""

question = st.text_input(
    "Enter your question:", 
    value=st.session_state.question_input, 
    key="question_input",
    placeholder="e.g., What are the management strategies for recurrent MDD?"
)

if st.button("Get Answer"):
    if question:
        with st.spinner("Searching for evidence and generating answer..."):
            try:
                qa_chain = load_rag_chain()
                result = qa_chain.invoke({"query": question})
                
                st.subheader("Generated Answer")
                st.markdown(result["result"])
                
                st.subheader("📚 Retrieved Sources")
                for source in result["source_documents"]:
                    with st.expander(f"Source: {os.path.basename(source.metadata['source'])}"):
                        st.markdown(f"**Content:**\n\n>{source.page_content.replace('\n', '\n> ')}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
