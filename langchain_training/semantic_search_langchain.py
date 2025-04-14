import os
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    # Get Mistral AI API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Please set the environment variable MISTRAL_API_KEY")

    pdf_path = os.getenv("PDF_PATH")

    # Initialize the Mistral AI embedding model
    embeddings = MistralAIEmbeddings(api_key=api_key, model="mistral-embed")

    # Load documents from a PDF file
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    all_splits = text_splitter.split_documents(documents)

    # Create an in-memory vector store and add document splits
    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=all_splits)

    # Perform a similarity search
    query = "What is Math"
    results = vector_store.similarity_search(query)

    # Print the top result
    if results:
        # print(results[0])
        print(results[0].page_content)

    else:
        print("No results found.")
