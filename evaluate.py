import time
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from sklearn.metrics.pairwise import cosine_similarity


test_questions = [
    # Question 1: Inspired by a patient with hypertension, tests knowledge from the clinical guideline (article1.pdf).
    "For a patient newly diagnosed with Stage 1 hypertension, what is the initial, non-pharmacological management strategy recommended by the 2017 ACC/AHA guidelines?",
    
    # Question 2: Tests specific fact retrieval from the Major Depressive Disorder article (article4.pdf).
    "What is the approximate heritability of Major Depressive Disorder as cited in the literature?",

    # Question 3: Inspired by a patient with RA, tests for a summary of a core concept from the Rheumatoid Arthritis article (article6.pdf).
    "What is the primary therapeutic goal in the management of Rheumatoid Arthritis?",

    # Question 4: Tests for a cause-and-effect relationship from the antibiotic resistance paper (article5.pdf).
    "What is a major factor contributing to the global crisis of antibiotic resistance?"
]

ground_truth_answers = [
    # Answer 1: Sourced from the ACC/AHA Hypertension Guideline (article1.pdf).
    "The initial and primary management strategy for Stage 1 hypertension involves lifestyle modifications, including weight loss, a heart-healthy diet such as the DASH diet, sodium restriction, and increased physical activity.",
    
    # Answer 2: Sourced from the Nature Reviews primer on MDD (article4.pdf).
    "The heritability of Major Depressive Disorder is estimated to be approximately 35-40%.",

    # Answer 3: Sourced from The Lancet seminar on Rheumatoid Arthritis (article6.pdf).
    "The primary goal of treating Rheumatoid Arthritis is to control disease activity to achieve clinical remission or low disease activity, thereby preventing joint damage and improving long-term function.",
    
    # Answer 4: Sourced from the paper on antibiotic resistance (article5.pdf).
    "A major factor contributing to antibiotic resistance is the overuse and misuse of antibiotics in both human medicine and agriculture."
]

DB_PATH = "vectorstores/db/"
MODEL_ID = "google/flan-t5-large"

def load_rag_pipeline():
    """Loads and returns the RAG chain and the embedding model."""
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    llm = HuggingFacePipeline.from_model_id(
        model_id=MODEL_ID,
        task="text2text-generation",
        model_kwargs={"temperature": 0.1, "max_length": 1024}
    )

    template = """
    You are an expert medical AI assistant. Your task is to provide a detailed, evidence-based answer to the user's question based ONLY on the provided context.
    INSTRUCTIONS:
    - Read the following context carefully.
    - Synthesize the information from all provided sources into a single, coherent paragraph.
    - If the context does not contain enough information to answer the question, you MUST state that clearly. Do not make up information.
    CONTEXT:
    {context}
    QUESTION:
    {question}
    DETAILED, EVIDENCE-BASED ANSWER:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain, embeddings


if __name__ == "__main__":
    rag_chain, embeddings = load_rag_pipeline()
    
    latencies = []
    similarities = []
    
    print("--- Running Evaluation ---")
    
    for i, question in enumerate(test_questions):
        print(f"\n--- Question {i+1} ---")
        print(f"Query: {question}")
        
 
        start_time = time.time()
        result = rag_chain.invoke({"query": question})
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)
        
        generated_answer = result["result"]
        ground_truth = ground_truth_answers[i]
        
  
        generated_embedding = embeddings.embed_query(generated_answer)
        ground_truth_embedding = embeddings.embed_query(ground_truth)
        
 
        similarity_score = cosine_similarity([generated_embedding], [ground_truth_embedding])[0][0]
        similarities.append(similarity_score)
        
        print(f"Generated Answer: {generated_answer}")
        print(f"Ground Truth Answer: {ground_truth}")
        print(f"Latency: {latency:.2f} seconds")
        print(f"Answer Similarity Score: {similarity_score:.4f}")

  
    average_latency = np.mean(latencies)
    average_similarity = np.mean(similarities)
    
    print("\n\n--- Average Evaluation Metrics ---")
    print(f"Average Latency: {average_latency:.2f} seconds")
    print(f"Average Answer Similarity: {average_similarity:.4f}")
    print("---------------------------------")