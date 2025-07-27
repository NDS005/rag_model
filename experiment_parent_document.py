import os
import io
import requests
import logging
from dotenv import load_dotenv

# --- Import LangChain components ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

# --- Configuration & Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global Objects ---
embedding_function = None

def initialize_models():
    """Initializes the embedding model."""
    global embedding_function
    if embedding_function is None:
        logging.info("Initializing embedding model...")
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        embedding_function = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        logging.info("Embedding model ready.")

def main():
    """
    This script demonstrates and tests the Parent Document Retriever strategy.
    """
    initialize_models()

    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    try:
        logging.info(f"Downloading document from: {document_url}")
        response = requests.get(document_url)
        response.raise_for_status()
        
        temp_pdf_path = "temp_document_for_parent_test.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download document: {e}")
        return

    # --- Load the Document ---
    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()

    # --- 1. Parent Document Retriever Setup ---
    # This store will hold the original parent documents
    docstore = InMemoryStore()

    # This is the splitter that will create the small child chunks
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    
    # This is the splitter that will create the larger parent chunks.
    # If None, it will use the original documents as parents.
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="parent_document_test", 
        embedding_function=embedding_function
    )

    logging.info("Setting up Parent Document Retriever...")
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # Add the documents to the retriever. This performs the chunking and embedding.
    retriever.add_documents(docs, ids=None)
    logging.info("Parent Document Retriever setup complete.")

    # --- 2. Run a Test Query ---
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    template = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    test_question = "What is the waiting period for cataract surgery?"
    logging.info(f"Running test query: '{test_question}'")
    
    # Let's inspect the retrieved documents
    retrieved_docs = retriever.invoke(test_question)
    print("\n--- Retrieved Parent Chunks for Context ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Context Chunk {i+1} ---")
        print(doc.page_content)
        print("-" * 20)

    answer = rag_chain.invoke(test_question)
    
    print("\n--- Test Query Result ---")
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")

    # Clean up the temporary file
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)

if __name__ == "__main__":
    main()
