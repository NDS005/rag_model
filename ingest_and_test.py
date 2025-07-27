import os
import glob
from dotenv import load_dotenv

# Correct, specialized import for the BGE model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Check if the API key is loaded correctly
hf_api_key = os.getenv("HF_TOKEN")
if not hf_api_key:
    print("Hugging Face API Key not found. Please check your .env file.")
    exit()

# Set the environment variable for HuggingFace
os.environ["HF_TOKEN"] = hf_api_key

DOCS_PATH = "documents/"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "global_policies"


def ingest_documents():
    """
    Ingests all PDF documents from the specified path into a Chroma vector store.
    """
    print("--- Starting Document Ingestion ---")
    
    pdf_files = glob.glob(os.path.join(DOCS_PATH, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{DOCS_PATH}'. Please add your documents.")
        return

    all_chunks = []
    
    for pdf_path in pdf_files:
        try:
            print(f"Processing document: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                separators=["\n\nSECTION", "\n\nPART", "\n\n[0-9]+\.", "\n\n", "\n", " ", ""],
                is_separator_regex=True,
            )
            chunks = text_splitter.split_documents(docs)
            
            doc_name = os.path.basename(pdf_path)
            for i, chunk in enumerate(chunks):
                page_number = chunk.metadata.get("page", 0) + 1
                chunk.metadata["clause_id"] = f"{doc_name}_p{page_number}_c{i}"
                chunk.metadata["source_document"] = doc_name
            
            all_chunks.extend(chunks)
            print(f"Split '{doc_name}' into {len(chunks)} chunks.")
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue

    if not all_chunks:
        print("No chunks were created. Halting ingestion.")
        return

    print("Initializing embedding model (BAAI/bge-large-en-v1.5)...")
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_function = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    print("Embedding model loaded.")

    print(f"Creating and persisting Vector DB at: {DB_PATH}")
    db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedding_function,
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
    )
    print("--- Ingestion Complete ---")


def test_retrieval():
    """
    Tests the retrieval by performing a similarity search on the existing vector store.
    """
    print("\n--- Starting Retrieval Test ---")

    if not os.path.exists(DB_PATH):
        print(f"Vector DB not found at '{DB_PATH}'. Please run the ingestion first.")
        return
        
    print("Initializing embedding model for testing...")
    model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    # Use the correct, specialized class here as well for consistency
    embedding_function = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    
    print("Loading existing vector store...")
    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

    sample_query = "What is the waiting period for pre-existing diseases?"
    print(f"\nPerforming search for query: '{sample_query}'")
    
    results = db.similarity_search(sample_query, k=4)

    if not results:
        print("No relevant documents found.")
        return

    print("\n--- Top 4 Relevant Chunks Found ---")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source Document: {doc.metadata.get('source_document')}")
        print(f"Clause ID: {doc.metadata.get('clause_id')}")
        print("-" * 20)
        print(doc.page_content)
        print("-" * 20)

if __name__ == "__main__":
    # This is now smarter. It will only ingest if the DB doesn't exist.
    if not os.path.exists(DB_PATH):
        ingest_documents()
    else:
        print("Database already exists. Skipping ingestion.")
    
    # Run a test query to see if it works
    test_retrieval()
