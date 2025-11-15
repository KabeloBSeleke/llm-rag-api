from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
# from langchain.chains.retrieval import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_huggingface import HuggingFaceEmbeddings
import os

LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "rag_collection")


class RAGService:
    """Manage RAG pipeline for data ingestion and querying. """

    def __init__(self, ollama_url: str):
        self.ollama_url = ollama_url
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None

    def load_embedding_model(self):
        """Loads the embedding model into memory."""
        # This is now a separate, controlled step
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        print("Embedding model loaded.")
    
    def init_vectorstore(self) -> str:
        """Initialize vector store from existing persisted data."""

        if self.embeddings is None:
            return "Embedding model not loaded. Cannot init vector store."
        
        # Check if verctor store path exists
        if not os.path.exists(VECTOR_STORE_PATH):
            return "Vector store path does not exist. Please ingest data first."
        
        # Load existing vector store
        self.vectorstore = Chroma(
            persistent_directory=VECTOR_STORE_PATH,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )
        return "Vector store initialized successfully."
    
    def init_qa_chain(self, llm_model_name: str = LLM_MODEL):
        # check if vector store is initialized
        if not self.vectorstore:
            raise ValueError("Vector store is not initialized. Please ingest data first.")
        
        # init LLM
        llm = Ollama(
            model=llm_model_name,
            base_url=self.ollama_url
        )

        # Create a retriever
        retriever = self.vectorstore.as_retriever()

        # Define the prompt template
        template = """Use the given context to answer the user's question.
            If you don't know the answer, say that you don't know.
            Context: {context}
            Question: {question}

            Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)

        # Create chain using LCEL
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.qa_chain = (
            {
                "context": retriever|format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # query method
    def query_rag(self, question: str) -> str:
        """Query the RAG system with a question."""
        if not self.qa_chain:
            raise RuntimeError("QA chain is not initialized. Please initialize it first.")

        # Invoke the QA chain with the question
        result = self.qa_chain.invoke(question)
        return result


