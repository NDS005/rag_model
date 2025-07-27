import os
import json
from dotenv import load_dotenv

# Using the latest, non-deprecated imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
load_dotenv()

DB_PATH = "./chroma_db"
COLLECTION_NAME = "global_policies"

def setup_retriever():
    """Initializes the vector store and retriever."""
    if not os.path.exists(DB_PATH):
        print(f"Vector DB not found at '{DB_PATH}'. Please run the ingestion script first.")
        return None

    print("Initializing embedding model for retrieval...")
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_function = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    print("Loading existing vector store...")
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )
    
    # Increased k to 7 to provide a wider context for multi-source citation
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
    print("Retriever setup complete.")
    return retriever

def format_docs(docs):
    """Formats the retrieved documents into a single string."""
    return "\n\n".join(
        f"Source: {doc.metadata.get('source_document', 'N/A')}, Clause ID: {doc.metadata.get('clause_id', 'N/A')}\nContent: {doc.page_content}"
        for doc in docs
    )

def main():
    retriever = setup_retriever()
    if not retriever:
        return

    llm = ChatGroq(model="llama3-70b-8192", temperature=0)

    # --- V4: MULTI-SOURCE & STRICT JSON PROMPT ---
    template = """
You are an expert insurance policy analyst. Your task is to analyze the provided context and user question and return a single, valid JSON object.

Follow these steps meticulously:
1.  **Analyze the User's Query:** Identify the core procedure, policy age, and any special conditions (like the cause of an injury).
2.  **Scan ALL Provided Context:** Read through every context chunk. Identify all rules, waiting periods, and exclusions that are relevant to the user's query, even if they come from different source documents.
3.  **Synthesize and Decide:** Formulate a single decision ('Approved', 'Rejected', 'Information') based on the combined evidence. A claim is rejected if it violates any single exclusion or waiting period.
4.  **Construct the JSON Response:**
    * `decision`: Your final decision.
    * `amount`: "N/A" unless a specific monetary value or percentage of sum is listed.
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

    print("\n--- Ready to answer questions ---")
    query = "My wife had a baby 45 days ago. Is her follow-up therapy session covered under the WellMother add-on? Is this covered?"
    
    print(f"\nQuerying the RAG system with: '{query}'")
    
    response_str = rag_chain.invoke(query)
    
    # Print and parse the JSON response
    try:
        # Pretty-print the JSON
        response_json = json.loads(response_str)
        print(json.dumps(response_json, indent=2))
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the response.")
        print("Raw Response:", response_str)

    print("\n\n--- RAG process complete. ---")

if __name__ == "__main__":
    main()
