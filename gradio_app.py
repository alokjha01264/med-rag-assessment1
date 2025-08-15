import gradio as gr
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA



DB_PATH = "vectorstores/db/"
MODEL_ID = "google/flan-t5-large"

def load_embedding_model():
    """Loads the sentence-transformers embedding model."""
    print("Loading embedding model...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={'device': 'cpu'})

def load_llm():
    """Loads the T5 language model."""
    print("Loading Language Model...")
    return HuggingFacePipeline.from_model_id(
        model_id=MODEL_ID,
        task="text2text-generation",
        model_kwargs={"temperature": 0.1, "max_length": 1024}
    )


print("Initializing RAG chain...")
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

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
print("RAG chain initialized successfully!")



def get_answer(question):
    """
    Takes a user question, invokes the RAG chain, and formats the output.
    """
    if not question:
        return "Please enter a question.", ""

    try:
    
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
   
        sources_text = "### 📚 Retrieved Sources\n\n"
        if result["source_documents"]:
            for source in result["source_documents"]:
                source_name = os.path.basename(source.metadata.get('source', 'Unknown'))
                sources_text += f"**Source:** `{source_name}`\n\n"
              
                page_content = "> " + source.page_content.replace("\n", "\n> ")
                sources_text += f"{page_content}\n\n---\n\n"
        else:
            sources_text = "No sources were retrieved for this answer."
            
        return answer, sources_text

    except Exception as e:
        return f"An error occurred: {e}", ""



example_questions = [
    "What are the 2017 AHA guideline recommendations for managing high blood pressure?",
    "What is the approximate heritability of Major Depressive Disorder?",
    "What is the primary therapeutic goal in the management of Rheumatoid Arthritis?"
]


with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🩺 Nervesparks Medical Literature RAG Assistant")
    gr.Markdown("This app is a Retrieval-Augmented Generation (RAG) system designed to assist healthcare professionals by answering questions based on a curated set of medical documents and synthetic patient data.")
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Enter your clinical question:",
                placeholder="e.g., What are the management strategies for recurrent MDD?",
                lines=3
            )
            submit_button = gr.Button("Get Answer", variant="primary")
            
            gr.Examples(
                examples=example_questions,
                inputs=question_input,
                label="Example Questions"
            )
            
        with gr.Column(scale=3):
            gr.Markdown("### Generated Answer")
            answer_output = gr.Markdown(value="Your answer will appear here...")
            
        
            with gr.Accordion("View Retrieved Sources", open=False):
                 sources_output = gr.Markdown(value="Source documents will appear here...")


    submit_button.click(
        fn=get_answer,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )

if __name__ == "__main__":
    app.launch()