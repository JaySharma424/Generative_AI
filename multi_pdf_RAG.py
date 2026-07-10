import os  # For interacting with the operating system

import streamlit as st  # for UI
from dotenv import load_dotenv
from PyPDF2 import PdfReader  # For reading PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # For splitting text into manageable chunks
from langchain_huggingface import HuggingFaceEmbeddings  # For generating embeddings locally
from langchain_google_genai import ChatGoogleGenerativeAI  # For chat-based interactions with Google AI
from langchain_community.vectorstores import FAISS  # For efficient similarity search with embeddings
from langchain_core.prompts import PromptTemplate  # For creating templates for prompts
from langchain_core.output_parsers import StrOutputParser  # To get plain text out of the model's response

load_dotenv()

FAISS_INDEX_DIR = "faiss_index"


def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            # scanned/image-only pages return None instead of "" -> guard with "or """
            text += page.extract_text() or ""
    return text


# split into chunks

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Create Embedding and store in FAISS Vector Database

def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_DIR)


# Build the LCEL chain: prompt -> Gemini -> plain string
# (load_qa_chain / langchain.chains.question_answering no longer exists in current LangChain,
#  so this replaces it with the same LCEL style you've already used with Gemini)

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible using the provided context.
    If the answer is not found in the context, say "Answer is not available in the context."

    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = prompt | model | StrOutputParser()
    return chain


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# process query

def chat_with_pdf(question):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # same embedding used for the vector DB
    db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)  # Vector DB gets loaded back
    docs = db.similarity_search(question)  # relevant CONTEXT
    context = format_docs(docs)
    chain = get_conversational_chain()  # LLM chain
    # Augmentation: retrieved context + question both go into the prompt
    answer = chain.invoke({"context": context, "question": question})
    return answer


# Streamlit Web App UI

st.title("💬 Chat with Your PDFs using Hugging Face + Gemini (RAG)")

if not os.getenv("GOOGLE_API_KEY"):
    st.warning("GOOGLE_API_KEY .env file me nahi mila. Gemini call fail hoga jab tak ye set na ho.")

# Sidebar: File Upload
with st.sidebar:
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # Process PDFs button
    if st.button("Process PDFs") and uploaded_files:
        with st.spinner("Processing..."):
            file_paths = []
            for file in uploaded_files:
                with open(file.name, "wb") as f:
                    f.write(file.read())
                    file_paths.append(file.name)

            raw_text = get_pdf_text(file_paths)
            chunks = get_text_chunks(raw_text)
            get_vector_store(chunks)
        st.success("✅ PDFs processed and indexed successfully!")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Main input area
st.header("Ask a Question")
question = st.text_input("Your Question")

# Submit question
if st.button("Submit") and question:
    if not os.path.exists(FAISS_INDEX_DIR):
        st.error("Pehle sidebar se PDF upload karke 'Process PDFs' dabao, index abhi tak nahi bana hai.")
    else:
        with st.spinner("Thinking..."):
            answer = chat_with_pdf(question)
        st.session_state.history.append((question, answer))

# Display Q&A
for q, a in st.session_state.history[::-1]:
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")