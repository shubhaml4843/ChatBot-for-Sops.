# -*- coding: utf-8 -*-


import os
from langchain.document_loaders import PyMuPDFLoader
from langchain_cohere import CohereEmbeddings  # Updated import
from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings, HuggingFaceEmbeddings
from transformers import pipeline
from langchain.prompts import PromptTemplate
import cohere

# Option to choose either Cohere or Hugging Face for embeddings and LLM
USE_COHERE = True  # Set to False if you prefer to use Hugging Face

if USE_COHERE:
    # Set up Cohere API key
    os.environ["COHERE_API_KEY"] = "ObzAJHH0UfJnsVXljRcSbgYi6KR5360PMtnC4E2O"

# Step 1: Load and Index the SOP Document
def load_and_index_documents(file_path):
    # Load document content from the uploaded PDF file
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    if USE_COHERE:
        # Generate embeddings using Cohere and store in a FAISS vector store
        embedding_model = CohereEmbeddings(model="small", user_agent="langchain-client")  # Use "small" or other sizes as needed
    else:
        # Generate embeddings using Hugging Face
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store embeddings in a FAISS vector store
    vector_store = FAISS.from_documents(documents, embedding_model)
    return vector_store

# Load and index the uploaded document
vector_store = load_and_index_documents('/content/purchase-manual-sop.pdf')

# Step 2: Define Question-Answering and Summarization Pipelines
if USE_COHERE:
    llm = cohere.Client(api_key=os.environ["COHERE_API_KEY"])

    def ask_sop_question(question):
        # Use Cohere's RAG chain for Q&A
        prompt = f"Answer the question based on the document:\nQuestion: {question}\nAnswer:"
        response = llm.generate(prompt=prompt, max_tokens=200)
        answer = response.generations[0].text.strip()

        # Generate a summary of the answer
        summary_prompt = f"Summarize the following answer:\n{answer}"
        summary_response = llm.generate(prompt=summary_prompt, max_tokens=50)
        summary = summary_response.generations[0].text.strip()
        return answer, summary

else:
    # Use Hugging Face Transformers for Q&A and summarization
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def ask_sop_question(question):
        # Perform question answering
        answer = qa_pipeline(question=question, context=" ".join([doc.page_content for doc in vector_store.index]))

        # Summarize the answer
        summary = summarizer(answer['answer'], max_length=100, min_length=55, do_sample=False)
        return answer['answer'], summary[0]['summary_text']

# Step 3: Test the Q&A System
question = "What is purpose of this sop"
answer ,summary = ask_sop_question(question)

# Display the results
print("\nanswer:\n" ,answer)
print("\nSummary:\n", summary)
