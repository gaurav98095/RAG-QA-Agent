## Create Your Agent and Load Data
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI


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
    
    def generate(
        self,
        query: str, 
        retrieved_docs: list, 
        llm_model=None, 
        prompt_template: str = None
    ):
        context = "\n".join(retrieved_docs)
        model = llm_model or OpenAI(temperature=0)
        prompt = prompt_template or (
            """
            You are a helpful assistant. Use the context below to answer the user's query. 
            Format your response strictly as a JSON object with the following structure:

            {
            "answer": "<a concise, complete answer to the user's query>",
            "citations": [
                "<relevant quoted snippet or summary from source 1>",
                "<relevant quoted snippet or summary from source 2>",
                ...
            ]
            }

            Only include information that appears in the provided context. Do not make anything up.
            Only respond in JSON — No explanations needed. Only use information from the context. If 
            nothing relevant is found, respond with: 

            {
            "answer": "No relevant information available.",
            "citations": []
            }


            Context:
            {context}

            Query:
            {query}
            """
        )
        prompt = prompt.format(context=context, query=query)
        return model(prompt)
    
    def answer(
        self, 
        query: str,
        llm_model=None, 
        prompt_template: str = None
    ):
        retrieved_docs = self.retrieve(query)
        generated_answer = self.generate(query, retrieved_docs, llm_model, prompt_template)
        return generated_answer, retrieved_docs


document_paths = ["neurolink-system.txt"]
query = "What are recent milestones for neurolink systems?"

agent = RAGAgent(document_paths)
retrieved_docs = agent.retrieve(query)
answer, retrieved_docs = agent.answer(query)
print(answer)

