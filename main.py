import os
import hashlib
import shutil
import logging
import asyncio
import requests
import tempfile
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# --- LangChain Imports ---
# Reverted to more reliable, pure-python loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    embedding_function = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    template =  """You are a universal document analysis assistant. Your primary function is to answer questions based strictly on the provided context.

    Core Principles:
    1.  Grounding: Extract information ONLY from the provided context. Do not use any external knowledge.
    2.  Completeness: If the context provides a rule (e.g., a financial limit), also look for its conditions and exceptions (e.g., a waiting period) to provide a complete answer.
    3.  Honesty: If the information is not available in the context, you MUST state: "This information is not available in the provided document." Do not speculate or infer.
    4.  Precision: Provide direct quotes and specific details from the context whenever possible to support your answer.

Context:
{context}

Question: {question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)
    logging.info("Models and prompt are ready.")
    yield
    logging.info("Application shutdown.")

app = FastAPI(title="HackRx RAG API", lifespan=lifespan)

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
        
        # Determine file type and create appropriate temporary file
        file_suffix = ".pdf"
        if ".docx" in document_url.lower():
            file_suffix = ".docx"
        # Add other types like .eml here if needed

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        # Choose the correct, reliable loader based on file type
        if file_suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif file_suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        else:
            # Fallback for unknown types
            loader = PyPDFLoader(tmp_path)

        documents = loader.load()
        os.remove(tmp_path)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function, persist_directory=cache_path)
        logging.info(f"Saved new vector store to cache: {cache_path}")
        return vectorstore.as_retriever(search_kwargs={"k": 7})

    except Exception as e:
        logging.exception(f"Error while processing document from {document_url}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

async def answer_question(question: str, retriever):
    try:
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        answer = await rag_chain.ainvoke(question)
        return answer.strip()
    except Exception as e:
        logging.error(f"Error in answer_question for '{question}': {str(e)}")
        return f"An error occurred while processing the question: {str(e)}"

@app.post("/hackrx/run", response_model=HackRxResponse)
async def process_documents_and_questions(
    request_data: HackRxRequest,
    token: HTTPAuthorizationCredentials = Depends(verify_token)
):
    try:
        retriever = get_retriever_for_url(request_data.documents)
        tasks = [answer_question(q, retriever) for q in request_data.questions]
        answers = await asyncio.gather(*tasks)
        return HackRxResponse(answers=answers)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"A critical error occurred in /hackrx/run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")