import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "rag_collection")


def run_ingestion():
    """Perform and persist data ingestion """

    # Check for persistance
    if os.path.exists(VECTOR_STORE_PATH):
        print("Data already ingested and vector store exists.")
        return
    print("Starting data ingestion...")

    # init embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    try:
        # load docs
        loader = TextLoader("./data/source-document.txt")
        documents = loader.load()

        # split docs
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)

        # create and persist vector store
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_PATH,
            collection_name=COLLECTION_NAME
        )
        # vectorstore.persist()
        print("Data ingestion completed and vector store persisted.")

    except Exception as e:
        print(f"Error during data ingestion: {str(e)}") 


if __name__ == "__main__":
    run_ingestion()

