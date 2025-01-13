import os
import streamlit as st
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import CohereEmbeddings, HuggingFaceEmbeddings
from transformers import pipeline
import cohere

# Option to choose either Cohere or Hugging Face for embeddings and LLM
USE_COHERE = True  # Set to False if you prefer to use Hugging Face

# Cohere API key setup
if USE_COHERE:
    os.environ["COHERE_API_KEY"] = "ObzAJHH0UfJnsVXljRcSbgYi6KR5360PMtnC4E2O"


# Token limit constants
COHERE_TOKEN_LIMIT = 4081
COHERE_SUMMARY_LIMIT = 512

# Step 1: Load and Index the SOP Document
def load_and_index_documents(uploaded_file):
    """Loads the uploaded file, processes it, and indexes the content in a FAISS vector store."""
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyMuPDFLoader(temp_file_path)
    documents = loader.load()

    if USE_COHERE:
        embedding_model = CohereEmbeddings(model="small", user_agent="langchain-client")
    else:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(documents, embedding_model)
    os.remove(temp_file_path)  # Clean up temporary file

    return vector_store, documents

# Step 2: Define Question-Answering and Summarization Logic
def ask_sop_question(question, documents):
    """Handles question answering and summarization based on the selected model."""
    document_content = " ".join([doc.page_content for doc in documents])

    if USE_COHERE:
        # Truncate document content if it exceeds the token limit
        truncated_content = document_content[:COHERE_TOKEN_LIMIT - len(question) - 50]

        # Question Answering
        prompt = f"Answer the question based on the document:\nQuestion: {question}\nDocument: {truncated_content}\nAnswer:"
        response = llm.generate(prompt=prompt, max_tokens=200)
        answer = response.generations[0].text.strip()

        # Summarization
        summary_prompt = f"Summarize the following answer:\n{answer}"
        summary_response = llm.generate(prompt=summary_prompt, max_tokens=50)
        summary = summary_response.generations[0].text.strip()
    else:
        context = document_content
        answer = qa_pipeline(question=question, context=context)
        summary = summarizer(answer['answer'], max_length=50, min_length=25, do_sample=False)

    return answer, summary

# Initialize models
if USE_COHERE:
    llm = cohere.Client(api_key=os.environ["COHERE_API_KEY"])
else:
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit interface
st.title("ChatBot for SOP")

# Upload file
uploaded_file = st.file_uploader("Upload your SOP PDF document", type="pdf")

if uploaded_file:
    try:
        st.info("Processing your document...")
        vector_store, documents = load_and_index_documents(uploaded_file)
        st.success("Document indexed successfully!")

        question = st.text_input("Ask a question about the SOP document:")

        if question:
            answer, summary = ask_sop_question(question, documents)
            st.subheader("Detailed Answer:")
            st.write(answer)

            st.subheader("Summary of the Answer:")
            st.write(summary)

        if st.button("Summarize the Entire SOP Document"):
            document_content = " ".join([doc.page_content for doc in documents])
            if USE_COHERE:
                truncated_content = document_content[:COHERE_SUMMARY_LIMIT]
                prompt = f"Summarize the following document:\n{truncated_content}"
                response = llm.generate(prompt=prompt, max_tokens=200)
                full_summary = response.generations[0].text.strip()
            else:
                full_summary = summarizer(document_content, max_length=200, min_length=50, do_sample=False)[0]['summary_text']

            st.subheader("Document Summary:")
            st.write(full_summary)
    except Exception as e:
        st.error(f"An error occurred: {e}")
