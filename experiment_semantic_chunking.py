import os
import io
import requests
import logging
from dotenv import load_dotenv

# --- Import LangChain components ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
# This is the new, experimental splitter
from langchain_experimental.text_splitter import SemanticChunker
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
    This script demonstrates and tests the Semantic Chunking strategy.
    """
    initialize_models()

    # Use the document URL from the hackathon example
    document_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    try:
        logging.info(f"Downloading document from: {document_url}")
        response = requests.get(document_url)
        response.raise_for_status()
        
        temp_pdf_path = "temp_document_for_semantic_test.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download document: {e}")
        return

    # --- Load the Document ---
    loader = PyPDFLoader(temp_pdf_path)
    documents = loader.load()
    # Combine all pages into a single document for the splitter
    full_text = "".join(doc.page_content for doc in documents)

    # --- 1. Semantic Chunking ---
    logging.info("Applying Semantic Chunker...")
    # The semantic chunker uses the embedding model to find sentence boundaries
    # that are semantically different, creating more coherent chunks.
    # 'breakpoint_threshold_type="percentile"' is a common starting point.
    semantic_splitter = SemanticChunker(
        embeddings=embedding_function, 
        breakpoint_threshold_type="percentile"
    )
    
    semantic_chunks = semantic_splitter.create_documents([full_text])
    
    logging.info(f"Created {len(semantic_chunks)} semantic chunks.")
    
    # --- Let's inspect a few chunks to see the difference ---
    print("\n--- Example Semantic Chunks ---")
    for i in range(min(3, len(semantic_chunks))):
        print(f"\n--- Chunk {i+1} (Length: {len(semantic_chunks[i].page_content)}) ---")
        print(semantic_chunks[i].page_content)
        print("-" * 20)

    # --- 2. Run a Test Query with these chunks ---
    logging.info("Creating a temporary in-memory vector store with semantic chunks...")
    vectorstore = Chroma.from_documents(semantic_chunks, embedding_function)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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
    
    answer = rag_chain.invoke(test_question)
    
    print("\n--- Test Query Result ---")
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")

    # Clean up the temporary file
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)

if __name__ == "__main__":
    main()
