import os
import io
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
import fitz  # PyMuPDF

# --- LangChain Imports ---
from langchain_chroma import Chroma
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
    embedding_function = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    template = "You are an expert insurance policy analyst. Answer the user's question based *only* on the provided context.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    prompt = ChatPromptTemplate.from_template(template)
    
    logging.info("Models and prompt are ready.")
    yield
    logging.info("Application shutdown.")

app = FastAPI(title="HackRx RAG API", lifespan=lifespan)

def intelligent_chunking(file_path: str) -> list[Document]:
    """
    Parses a PDF using PyMuPDF to understand its layout (headings, lists)
    and creates context-rich chunks to prevent orphaned clauses.
    """
    logging.info("Starting intelligent, layout-aware chunking...")
    doc = fitz.open(file_path)
    
    # Heuristics to identify headings (e.g., larger font size, bold)
    # These might need tuning for different document styles
    HEADING_FONT_SIZE_THRESHOLD = 11.0 
    
    # A list to hold our structured, context-rich chunks
    structured_chunks = []
    current_heading = ""
    current_content = ""

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        font_size = span["size"]
                        is_bold = "bold" in span["font"].lower()

                        # Check if this span is a heading
                        if font_size > HEADING_FONT_SIZE_THRESHOLD or is_bold:
                            # If we have content under a previous heading, save it as a chunk
                            if current_heading and current_content:
                                structured_chunks.append(Document(page_content=f"{current_heading}\n{current_content.strip()}"))
                            
                            # Start a new section
                            current_heading = text
                            current_content = ""
                        else:
                            # Append content to the current section
                            current_content += text + " "
    
    # Add the last processed chunk
    if current_heading and current_content:
        structured_chunks.append(Document(page_content=f"{current_heading}\n{current_content.strip()}"))

    # Also, get the full text for a fallback standard chunking
    full_text = "\n".join([page.get_text() for page in doc])
    standard_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    standard_chunks = standard_splitter.create_documents([full_text])

    # Combine both strategies for robustness
    final_chunks = structured_chunks + standard_chunks
    logging.info(f"Created {len(final_chunks)} intelligent chunks.")
    return final_chunks


def get_retriever_for_url(document_url: str):
    url_hash = hashlib.md5(document_url.encode()).hexdigest()
    cache_path = os.path.join(CACHE_DIR, url_hash)

    if os.path.exists(cache_path):
        logging.info(f"Cache hit. Loading vector store from: {cache_path}")
        vectorstore = Chroma(persist_directory=cache_path, embedding_function=embedding_function)
        return vectorstore.as_retriever(search_kwargs={"k": 5})

    logging.info(f"Cache miss. Processing document from: {document_url}")
    
    try:
        response = requests.get(document_url)
        response.raise_for_status()
        
        temp_file_path = f"temp_{url_hash}.pdf"
        with open(temp_file_path, "wb") as f:
            f.write(response.content)

        # --- USE OUR NEW INTELLIGENT CHUNKING FUNCTION ---
        chunks = intelligent_chunking(temp_file_path)
        os.remove(temp_file_path)

        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embedding_function, persist_directory=cache_path
        )
            
        logging.info(f"Saved new vector store to cache: {cache_path}")
        return vectorstore.as_retriever(search_kwargs={"k": 5})
        
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
