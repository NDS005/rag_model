import os
import io
import hashlib
import shutil
import logging
import asyncio
import pickle
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
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
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
        logging.warning("Clearing existing document cache to ensure a fresh start.")
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    template = "You are an expert insurance policy analyst. Answer the user's question based *only* on the provided context.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    prompt = ChatPromptTemplate.from_template(template)
    
    logging.info("Models and prompt are ready.")
    yield
    logging.info("Application shutdown.")

app = FastAPI(title="HackRx RAG API", lifespan=lifespan)

def get_retriever_for_url(document_url: str):
    url_hash = hashlib.md5(document_url.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, url_hash)
    docstore_path = os.path.join(cache_path, "docstore.pkl")

    # --- CACHE HIT LOGIC (FIXED) ---
    if os.path.exists(cache_path) and os.path.exists(docstore_path):
        logging.info(f"Cache hit. Loading vector store and docstore from: {cache_path}")
        vectorstore = Chroma(persist_directory=cache_path, embedding_function=embedding_function)
        
        with open(docstore_path, "rb") as f:
            stored_documents = pickle.load(f)
        docstore = InMemoryStore()
        docstore.mset(list(stored_documents.items()))

        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        return ParentDocumentRetriever(vectorstore=vectorstore, docstore=docstore, child_splitter=child_splitter)

    # --- CACHE MISS LOGIC (FIXED) ---
    logging.info(f"Cache miss. Processing document from: {document_url}")
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    os.makedirs(cache_path)
    
    try:
        response = requests.get(document_url)
        response.raise_for_status()
        
        temp_file_path = f"temp_{url_hash}"
        with open(temp_file_path, "wb") as f:
            f.write(response.content)

        if document_url.lower().endswith('.pdf'):
            loader = PyPDFLoader(temp_file_path)
        elif document_url.lower().endswith('.docx'):
            loader = Docx2txtLoader(temp_file_path)
        else:
            loader = PyPDFLoader(temp_file_path)
        
        docs = loader.load()
        os.remove(temp_file_path)

        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        
        # --- ACCURACY FIX: Added custom separators to the parent splitter ---
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=[
                "\n\niv. ", "\n\niii. ", "\n\nii. ", "\n\ni. ", # Roman numerals for lists
                "\n\nf) ", "\n\ne) ", "\n\nd) ", "\n\nc) ", "\n\nb) ", "\n\na) ", # Lettered lists
                "\n\n", "\n", " ", ""
            ]
        )
        vectorstore = Chroma(collection_name=url_hash, embedding_function=embedding_function, persist_directory=cache_path)
        docstore = InMemoryStore()

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore, docstore=docstore, child_splitter=child_splitter, parent_splitter=parent_splitter
        )
        retriever.add_documents(docs, ids=None)
        
        with open(docstore_path, "wb") as f:
            pickle.dump(docstore.store, f)
            
        logging.info(f"Saved new vector store and docstore to cache: {cache_path}")
        return retriever
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
