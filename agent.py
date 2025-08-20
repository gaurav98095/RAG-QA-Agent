## Create Your Agent and Load Data
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

class RAGAgent:
    def __init__(
        self,
        document_paths: list,
        embedding_model=None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store_class=FAISS,
        k: int = 2
    ):
        self.document_paths = document_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.vector_store_class = vector_store_class
        self.k = k
        self.vector_store = self._load_vector_store()
    
    def _load_vector_store(self):
        documents = []
        for document_path in self.document_paths:
            with open(document_path, "r", encoding="utf-8") as file:
                raw_text = file.read()
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            documents.extend(splitter.create_documents([raw_text]))

        return self.vector_store_class.from_documents(documents, self.embedding_model)
    
    def retrieve(self, query: str):
        docs = self.vector_store.similarity_search(query, k=self.k)
        context = [doc.page_content for doc in docs]
        return context
    
document_paths = ["neurolink-system.txt"]
agent = RAGAgent(document_paths)
retrieved_docs = agent.retrieve("How many blood tests can you perform and how much blood do you need?")
print(retrieved_docs)


