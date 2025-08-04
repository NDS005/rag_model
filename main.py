import os
import shutil
import logging
import asyncio
import requests  # ✅ Added for downloading PDFs from URL
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

EXPECTED_API_KEY = os.getenv("HACKRX_API_KEY")
auth_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not EXPECTED_API_KEY or credentials.scheme != "Bearer" or credentials.credentials != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return credentials

# Request and Response Models
class HackRxRequest(BaseModel):
    documents: str  # URL to the PDF
    questions: list[str]

class HackRxResponse(BaseModel):
    answers: list[str]

# Globals
CACHE_DIR = "./document_cache"
embedding_function = None
llm = None

# Lifespan setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_function, llm
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/rag", response_model=HackRxResponse)
async def rag_endpoint(request: HackRxRequest, token: HTTPAuthorizationCredentials = Depends(verify_token)):
    pdf_url = request.documents

    try:
        response = requests.get(pdf_url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download PDF from the URL")
        
        # Save temporarily
        local_pdf_path = os.path.join(CACHE_DIR, "document.pdf")
        with open(local_pdf_path, "wb") as f:
            f.write(response.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading PDF: {str(e)}")

    # 🔥 Use PDFPlumberLoader
    loader = PDFPlumberLoader(local_pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(chunks, embedding=embedding_function)

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 15, "lambda_mult": 0.25}
    )

    prompt = PromptTemplate(
        template="""You are a helpful AI assistant. Use the provided context to answer the user's question.
If the context does not fully answer the question, you may use your own knowledge as needed — but avoid hallucinations.

Always provide clear, accurate, and helpful responses.
Avoid saying things like "this information is not available in the context."

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    answers = [qa_chain.run(q) for q in request.questions]
    return HackRxResponse(answers=answers)