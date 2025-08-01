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
from langchain_community.document_loaders import UnstructuredFileLoader
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_function, llm, prompt
    logging.info("Application startup: Initializing models...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_function = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    template = """You are a universal document analysis assistant... (your prompt here)"""
    prompt = ChatPromptTemplate.from_template(template)
    logging.info("Models and prompt are ready.")
    yield
    logging.info("Application shutdown.")

app = FastAPI(title="HackRx Generalized RAG API", lifespan=lifespan)

def download_and_load_document(document_url: str):
    try:
        response = requests.get(document_url)
        response.raise_for_status()
        
        # Determine the correct suffix
        file_suffix = ".pdf" # Default
        if ".docx" in document_url:
            file_suffix = ".docx"
        # Add more types as needed

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        loader = UnstructuredFileLoader(tmp_path)
        documents = loader.load()
        os.remove(tmp_path)
        return documents
    except Exception as e:
        logging.exception("Error during file download or loading")
        raise ValueError(f"Error downloading or loading document: {e}")

def get_retriever_for_url(document_url: str):
    try:
        documents = download_and_load_document(document_url)
        if not documents:
            raise ValueError("No documents returned by the loader.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        if not splits:
            raise ValueError("Document could not be split into chunks.")

        url_hash = hashlib.md5(document_url.encode()).hexdigest()
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function, collection_name=f"docs_{url_hash}")
        return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 7, "fetch_k": 15})
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
        # Use ainoke for true async behavior with LangChain
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
        
        # This is a more robust way to run async tasks
        answer_tasks = [answer_question(q, retriever) for q in request_data.questions]
        answers = await asyncio.gather(*answer_tasks)
        
        return HackRxResponse(answers=answers)
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"A critical error occurred in /hackrx/run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")