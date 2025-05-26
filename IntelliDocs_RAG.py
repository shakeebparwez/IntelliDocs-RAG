import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template
import re

# Clean <think> tags
def clean_output(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# Extract PDF text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Create vector store
def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Custom Question Answering
def answer_question(llm, retriever, question):
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    template = """
You are a helpful assistant for answering questions based on documents.
Use the context provided to answer the user's question **briefly and accurately**.
Avoid unnecessary repetition or internal thoughts.

Context:
{context}

Question:
{question}

Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    formatted_prompt = prompt.format(context=context, question=question)

    raw_answer = llm.invoke(formatted_prompt)
    cleaned_answer = clean_output(raw_answer)
    return cleaned_answer

# Handle user input
def handle_userinput(user_question):
    llm = st.session_state.llm
    retriever = st.session_state.retriever
    answer = answer_question(llm, retriever, user_question)

    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("assistant", answer))

    for role, message in st.session_state.chat_history:
        if role == "user":
            st.write(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)

# Main app
def main():
    load_dotenv()
    st.set_page_config(page_title="IntelliDocs-RAG", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("📄 IntelliDocs-RAG: Chat with your PDFs")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.vectorstore is not None:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.vectorstore = vectorstore
                st.session_state.llm = OllamaLLM(model="deepseek-r1:1.5b")
                st.session_state.retriever = vectorstore.as_retriever()
                st.session_state.chat_history = []

if __name__ == '__main__':
    main()
