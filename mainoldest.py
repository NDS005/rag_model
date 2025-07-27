import os
import json
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import the RAG chain components from our query script
from query_rag import setup_retriever, format_docs
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Global RAG Chain ---
# This will be initialized during the application's lifespan startup.
rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI app.
    Loads the RAG chain on startup.
    """
    global rag_chain
    print("Application startup: Loading RAG chain...")
    
    retriever = setup_retriever()
    if not retriever:
        raise RuntimeError("Failed to setup retriever. Make sure your vector DB is created.")

    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    
    # Using the latest, most robust prompt for multi-source and strict JSON
    template = """
You are an expert insurance policy analyst. Your task is to analyze the provided context and user question and return a single, valid JSON object.

Follow these steps meticulously:
1.  **Analyze the User's Query:** Identify the core procedure, policy age, and any special conditions (like the cause of an injury).
2.  **Scan ALL Provided Context:** Read through every context chunk. Identify all rules, waiting periods, and exclusions that are relevant to the user's query, even if they come from different source documents.
3.  **Synthesize and Decide:** Formulate a single decision ('Approved', 'Rejected', 'Information') based on the combined evidence. A claim is rejected if it violates any single exclusion or waiting period.
4.  **Construct the JSON Response:**
    * `decision`: Your final decision.
    * `amount`: "N/A" unless a specific monetary value is approved.
    * `justification`: This MUST be a list of objects. Each object must contain a `quote` (a direct, short quote from the context) and the corresponding `clause_id`. You MUST include an object for every piece of evidence you used from the context.

Your final output MUST be ONLY the JSON object. Do not include any text like "Here is the JSON response:" or any explanations after the JSON object.

CONTEXT:
----------------
{context}
----------------

USER'S QUESTION:
{question}
----------------

JSON RESPONSE:
"""
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG Chain is ready.")
    yield
    # Cleanup logic would go here if needed
    print("Application shutdown.")


# --- App Initialization with Lifespan Event Handler ---
app = FastAPI(
    title="Insurance Policy RAG API",
    description="An API to query insurance policy documents using Retrieval-Augmented Generation.",
    lifespan=lifespan
)

# --- Pydantic Models for Request/Response Body ---
class QueryRequest(BaseModel):
    question: str
    user_id: str | None = None # Optional user_id for user-specific docs

# --- API Endpoints ---
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Receives a question and returns the RAG system's answer.
    """
    if not rag_chain:
        return JSONResponse(status_code=503, content={"error": "RAG chain is not ready."})

    try:
        response_str = rag_chain.invoke(request.question)
        # Since the prompt is now strict, we can directly parse the string
        response_json = json.loads(response_str)
        return JSONResponse(content=response_json)
    except json.JSONDecodeError:
        return JSONResponse(status_code=500, content={"error": "Failed to decode JSON from LLM response.", "raw_response": response_str})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload")
async def upload_document(user_id: str = Form(...), file: UploadFile = File(...)):
    """
    A placeholder endpoint for uploading user-specific documents.
    (This requires the full user-specific ingestion logic to be built out)
    """
    # For now, we'll just save the file to show it works
    upload_dir = f"user_documents/{user_id}"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # In a real implementation, you would now trigger the ingestion
    # process for this file into a user-specific ChromaDB collection.
    
    return {"message": f"File '{file.filename}' uploaded for user '{user_id}'. Ingestion needs to be implemented."}

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API. Use the /docs endpoint to see the API documentation."}

# To run this app:
# 1. Make sure you have uvicorn and python-multipart installed: pip install uvicorn python-multipart
# 2. Run in your terminal: uvicorn main:app --reload
