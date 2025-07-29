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
from pydantic import BaseModel

# --- LangChain Imports ---
# 1. UPGRADE: Using a more powerful, layout-aware document loader
from langchain_community.document_loaders import UnstructuredURLLoader
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
    """Verifies the API key provided in the request header."""
    if not EXPECTED_API_KEY or credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return credentials

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    """Defines the structure of the incoming API request."""
    documents: str
    questions: list[str]

class HackRxResponse(BaseModel):
    """Defines the structure of the API response."""
    answers: list[str]

# --- Global Objects ---
embedding_function = None
llm = None
prompt = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models and resources on application startup."""
    global embedding_function, llm, prompt
    logging.info("Application startup: Initializing models...")

    # Initialize a powerful embedding model for better semantic understanding
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    
    # Initialize the Large Language Model from Groq
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    
    # A universal prompt designed to handle any document type by focusing on core principles
    template = """You are a universal document analysis assistant. Your primary function is to answer questions based strictly on the provided context.

    Core Principles:
    1.  *Grounding:* Extract information ONLY from the provided context. Do not use any external knowledge.
    2.  *Completeness:* If the context provides a rule (e.g., a financial limit), also look for its conditions and exceptions (e.g., a waiting period) to provide a complete answer.
    3.  *Honesty:* If the information is not available in the context, you MUST state: "This information is not available in the provided document." Do not speculate or infer.
    4.  *Precision:* Provide direct quotes and specific details from the context whenever possible to support your answer.

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    logging.info("Models and prompt are ready.")
    yield
    logging.info("Application shutdown.")

app = FastAPI(title="HackRx Generalized RAG API", lifespan=lifespan)

def get_retriever_for_url(document_url: str):
    """
    Creates a retriever for a given document URL using an advanced, structure-aware process.
    """
    try:
        # 1. THE UPGRADE: Use UnstructuredURLLoader for robust, layout-aware parsing.
        # This is better than PyPDFLoader as it can understand tables, titles, and lists.
        loader = UnstructuredURLLoader(urls=[document_url], strategy="fast", ssl_verify=False)
        documents = loader.load()
        
        # 2. THE FIX: Implement a smarter, rule-based chunking strategy.
        # These separators are designed to respect the logical structure of policy/legal documents,
        # preventing "orphan clauses" by keeping rules and their headers together.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=[
                "\n\n\n",                # Major section breaks
                "\n\n",                  # Paragraphs
                "\n",                    # Line breaks
                ". ",                    # Sentences
                " ",                     # Words
                r"\n\d{1,2}\.\d{1,2}\.\s", # e.g., "\n5.15. "
                r"\n\d{1,2}\.\s",         # e.g., "\n4. "
                r"\n[a-z]\)",             # e.g., "\na)"
            ]
        )
        
        splits = text_splitter.split_documents(documents)
        
        if not splits:
            raise ValueError("Document could not be split into chunks. It might be empty or unreadable.")

        # Generate a unique, safe collection name for ChromaDB
        url_hash = hashlib.md5(document_url.encode()).hexdigest()
        collection_name = f"docs_{url_hash}"

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            collection_name=collection_name
        )
        
        # Retrieve a slightly larger context window (k=7) to ensure all relevant clauses
        # are sent to the LLM, allowing it to form a complete and accurate answer.
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 7, "fetch_k": 15}
        )
        
        logging.info(f"Successfully created retriever for document with {len(splits)} chunks")
        return retriever
        
    except Exception as e:
        logging.error(f"Error creating retriever for URL {document_url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

def answer_question(question: str, retriever):
    """Answers a single question using the provided RAG chain."""
    try:
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        answer = rag_chain.invoke(question)
        return answer.strip()
        
    except Exception as e:
        logging.error(f"Error in answer_question for '{question}': {str(e)}")
        return f"An error occurred while processing the question: {str(e)}"

@app.post("/hackrx/run", response_model=HackRxResponse)
async def process_documents_and_questions(
    request_data: HackRxRequest,
    token: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """
    Main API endpoint. Ingests a document, creates a retriever,
    and answers a list of questions based on the document's content.
    """
    try:
        logging.info(f"Processing document from URL: {request_data.documents}")
        logging.info(f"Answering {len(request_data.questions)} questions.")
        
        retriever = get_retriever_for_url(request_data.documents)
        
        # Process questions in parallel for better performance
        answer_tasks = [
            asyncio.to_thread(answer_question, q, retriever) for q in request_data.questions
        ]
        answers = await asyncio.gather(*answer_tasks)
        
        logging.info("All questions processed successfully.")
        return HackRxResponse(answers=answers)
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"A critical error occurred in /hackrx/run: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

if __name__ == "_main_":
    import uvicorn
    # The uvicorn server expects the filename to be 'main' and the app instance to be 'app'
    # e.g., uvicorn.run("main:app", ...)
    # Since you saved the file as gang.py, you would run it from the terminal as:
    # uvicorn gang:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8001)