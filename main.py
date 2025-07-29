import os
import hashlib
import shutil
import logging
import asyncio
import re
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import requests

# --- LangChain Imports ---
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- Configuration & Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Authentication ---
EXPECTED_API_KEY = os.getenv("HACKRX_API_KEY")
auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not EXPECTED_API_KEY or credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return credentials

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: str
    questions: list[str]

class HackRxResponse(BaseModel):
    answers: list[str]

# --- Global Objects ---
embedding_function = None
llm = None
prompt = None
CACHE_DIR = "./document_cache"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_function, llm, prompt
    logging.info("Application startup: Initializing models...")
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    template = """
    You are a universal document analysis assistant capable of understanding and analyzing any type of document content. Answer questions based strictly on the provided context.

Core Principles:
1. Extract information ONLY from the provided context
2. If information is not available in the context, state: "This information is not available in the provided document(s)"
3. Provide direct quotes and specific references when possible
4. Maintain accuracy and avoid speculation
5. Adapt your response style to match the document type and question complexity

For any document type (academic, technical, legal, historical, scientific, policy, etc.):
- Identify key facts, figures, dates, and relationships
- Explain concepts clearly and concisely
- Highlight relevant sections that support your answer
- Structure your response logically
    \n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""
    prompt = ChatPromptTemplate.from_template(template)
    
    logging.info("Models and prompt are ready.")
    yield
    logging.info("Application shutdown.")

app = FastAPI(title="HackRx RAG API", lifespan=lifespan)

def preprocess_text_for_waiting_periods(text: str) -> str:
    """
    Injects waiting period context into list items for better retrieval.
    """
    logging.info("Starting context-aware pre-processing...")
    lines = text.split('\n')
    processed_lines = []
    current_context = ""

    # Regex to find the waiting period headings
    patterns = {
        "90 Days Waiting Period": re.compile(r"i\.\s+90\s+Days\s+Waiting\s+Period", re.IGNORECASE),
        "One year waiting period": re.compile(r"ii\.\s+One\s+year\s+waiting\s+period", re.IGNORECASE),
        "Two years waiting period": re.compile(r"iii\.\s+Two\s+years\s+waiting\s+period", re.IGNORECASE),
        "Three years waiting period": re.compile(r"iv\.\s+Three\s+years\s+waiting\s+period", re.IGNORECASE),
    }

    for line in lines:
        found_new_context = False
        for context, pattern in patterns.items():
            if pattern.search(line):
                current_context = context
                found_new_context = True
                break
        
        # If it's a list item and we have a context, inject it
        if re.match(r'^\s*[a-z]\.\s+', line) and current_context:
            # Clean up the list item before prepending context
            clean_line = line.strip().split('.', 1)[-1].strip()
            processed_lines.append(f"Waiting Period Context: {current_context}; Procedure: {clean_line}")
        else:
            processed_lines.append(line)
            if found_new_context:
                current_context = "" # Reset context after the heading line itself is processed

    logging.info("Pre-processing complete.")
    return "\n".join(processed_lines)


def get_retriever_for_url(document_url: str):
    url_hash = hashlib.md5(document_url.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, url_hash)

    if os.path.exists(cache_path):
        logging.info(f"Cache hit. Loading vector store from: {cache_path}")
        vectorstore = Chroma(persist_directory=cache_path, embedding_function=embedding_function)
        return vectorstore.as_retriever(search_kwargs={"k": 7})

    logging.info(f"Cache miss. Processing document from: {document_url}")
    
    try:
        response = requests.get(document_url)
        response.raise_for_status()
        
        temp_file_path = f"temp_{url_hash}"
        with open(temp_file_path, "wb") as f:
            f.write(response.content)

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        os.remove(temp_file_path)

        # --- NEW: Pre-process the text before chunking ---
        full_text = "\n".join([doc.page_content for doc in docs])
        processed_text = preprocess_text_for_waiting_periods(full_text)
        processed_docs = [Document(page_content=processed_text)]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(processed_docs)

        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embedding_function, persist_directory=cache_path
        )
            
        logging.info(f"Saved new vector store to cache: {cache_path}")
        return vectorstore.as_retriever(search_kwargs={"k": 7})
        
    except Exception as e:
        logging.error(f"Failed to process document for caching: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

async def answer_question(question: str, retriever):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return await rag_chain.ainvoke(question)

@app.post("/hackrx/run", response_model=HackRxResponse)
async def process_documents_and_questions(
    request_data: HackRxRequest,
    token: HTTPAuthorizationCredentials = Depends(verify_token)
):
    try:
        retriever = get_retriever_for_url(request_data.documents)
        
        logging.info(f"Processing {len(request_data.questions)} questions in parallel...")
        tasks = [answer_question(q, retriever) for q in request_data.questions]
        answers = await asyncio.gather(*tasks)
        
        stripped_answers = [ans.strip() for ans in answers]
        logging.info("All questions answered successfully.")
        return {"answers": stripped_answers}
    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
