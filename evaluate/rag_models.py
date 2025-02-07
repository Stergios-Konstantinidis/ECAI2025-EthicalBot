from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable, RunnableSequence
from langchain.schema import StrOutputParser
import logging
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field
from typing import Any, List, Optional
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
import pickle
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


logging.basicConfig(level=logging.ERROR)

######## Naive RAG ########

class NaiveRAG:
    def __init__(self, chroma_path, embedding_function, llm_model, prompt_template):
        self.chroma_path = chroma_path
        try:
            self.embedding_model = embedding_function
        except Exception as e:           
            logging.error(f"Error initializing embedding model: {e}")
            exit(1)
        try:
            self.llm_model = llm_model     
        except Exception as e:
            logging.error(f"Error initializing llm: {e}")    
            exit(1)
        
        self.db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_model)
        self.retriever = self.db.as_retriever(search_kwargs={"k": 3})
        self.prompt_template = ChatPromptTemplate.from_template(prompt_template)
        
        
    def format_docs(self, results):
        return  "\n\n---\n\n".join([doc.page_content for doc in results])
    
    def rag_chain(self):
        rag_chain = (
            {
                "context": self.retriever | self.format_docs,   # Retrieve top 3 docs and format them
                "question": RunnablePassthrough()     # Pass the question through without modification
            }
            | self.prompt_template                        # Format the prompt with context and question
            | self.llm_model                                    # Pass the prompt to the LLM for response generation
            | StrOutputParser()                      # Parse the string output from the LLM
        )
        return rag_chain
        

######## Parent Retriever ########

class ParentDocumentRetriever(BaseRetriever):
    """Retriever for retrieving related documents based on document ID and sequence."""

    client: Chroma = Field(...)          # Define `client` with Field so that Pydantic can recognize it as a field.
    window_size: int = Field(default=2)  # Add type hints and defaults for other fields
    k: int = Field(default=3)            # Add type hints and defaults for other fields    

    def __init__(self, client: Chroma, window_size: int = 2, k: int = 3):
        super().__init__(client=client, window_size=window_size, k=k)  # Initialize Pydantic fields
        self.client = client
        self.window_size = window_size
        self.k = k

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve top documents based on query similarity and neighboring sequence numbers."""
        results = self.client.similarity_search(query=query, k=self.k)
        docs_to_return = []
        
        for result in results:
            doc_id = result.metadata["document_id"]
            seq_num = result.metadata["sequence_number"]
            ids_window = [seq_num + i for i in range(-self.window_size, self.window_size + 1, 1)]
            
            expr = {
                "$and": [
                    {"document_id": {"$eq": doc_id}},
                    {"sequence_number": {"$gte": ids_window[0]}},
                    {"sequence_number": {"$lte": ids_window[-1]}},
                ]
            }
            
            res = self.client.get(where=expr)
            texts, metadatas = res["documents"], res["metadatas"]
            
            combined_text = " ".join(texts)
            combined_metadata = {
                "document_id": doc_id,
                "sequence_numbers": [meta["sequence_number"] for meta in metadatas],
                "source": result.metadata.get("source"),
                "page": result.metadata.get("page")
            }
            
            docs_to_return.append(
                Document(
                    page_content=combined_text,
                    metadata=combined_metadata
                )
            )
            
        return docs_to_return

    # Optionally, we can implement an asynchronous method for Trulens
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Async retrieval of relevant documents."""
        return self._get_relevant_documents(query, run_manager=run_manager)



class ParentRetriver:
    def __init__(self, chroma_path, embedding_function, llm_model, prompt_template,k:int=3, window_size:int=2): 
        self.chroma_path = chroma_path
        try:
            self.embedding_model = embedding_function
        except Exception as e:           
            logging.error(f"Error initializing embedding model: {e}")
            exit(1)
        try:
            self.llm_model = llm_model     
        except Exception as e:
            logging.error(f"Error initializing llm model: {e}")    
            exit(1)
        
        self.db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_model)
        self.retriever = ParentDocumentRetriever(client=self.db, window_size=window_size, k=k)
        
        self.prompt_template = ChatPromptTemplate.from_template(prompt_template)
        
    def format_docs(self, results):
        return  "\n\n---\n\n".join([doc.page_content for doc in results])
    
    def rag_chain(self):
        rag_chain = (
            {
                "context": self.retriever | self.format_docs,   # Retrieve top 3 docs and format them
                "question": RunnablePassthrough()     # Pass the question through without modification
            }
            | self.prompt_template                        # Format the prompt with context and question
            | self.llm_model                                    # Pass the prompt to the LLM for response generation
            | StrOutputParser()                      # Parse the string output from the LLM
        )
        return rag_chain
    
    
    
    ######## Multi-Query Retriever ########
    
    
class MultiQuery:
    def __init__(self, chroma_path, embedding_function, llm_model, prompt_template):
        self.chroma_path = chroma_path
        try:
            self.embedding_model = embedding_function
        except Exception as e:           
            logging.error(f"Error initializing embedding model: {e}")
            exit(1)
        try:
            self.llm_model = llm_model     
        except Exception as e:
            logging.error(f"Error initializing llm model: {e}")    
            exit(1)
        
        self.db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_model)
        self.retriever = MultiQueryRetriever.from_llm(retriever=self.db.as_retriever(search_kwargs={"k": 3}), llm=self.llm_model)
        self.prompt_template = ChatPromptTemplate.from_template(prompt_template)
        
    def format_docs(self, results):
        return  "\n\n---\n\n".join([doc.page_content for doc in results])

    def rag_chain(self):
        rag_chain = (
            {
                "context": self.retriever | self.format_docs,       # Retrieve top 3 docs and format them
                "question": RunnablePassthrough()                   # Pass the question through without modification
            }
            | self.prompt_template                                  # Format the prompt with context and question
            | self.llm_model                                        # Pass the prompt to the LLM for response generation
            | StrOutputParser()                                     # Parse the string output from the LLM
        )
        return rag_chain
    
    
    
######## Parent Retriever with Multi-Query ########

class ParentDocumentRetrieverwithMultiQuery:
    def __init__(self, chroma_path, embedding_function, llm_model, prompt_template, k:int=3, window_size:int=2):
        self.chroma_path = chroma_path
        try:
            self.embedding_model = embedding_function
        except Exception as e:           
            logging.error(f"Error initializing embedding model: {e}")
            exit(1)
        try:
            self.llm_model = llm_model     
        except Exception as e:
            logging.error(f"Error initializing llm model: {e}")    
            exit(1)
        
        self.db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_model)
        self.retriever = MultiQueryRetriever.from_llm(retriever=ParentDocumentRetriever(client=self.db, window_size=window_size, k=k), llm=self.llm_model)
        
        self.prompt_template = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(self, results):
        return  "\n\n---\n\n".join([doc.page_content for doc in results])

    def rag_chain(self):
        rag_chain = (
            {
                "context": self.retriever | self.format_docs,       # Retrieve top 3 docs and format them
                "question": RunnablePassthrough()                   # Pass the question through without modification
            }
            | self.prompt_template                                  # Format the prompt with context and question
            | self.llm_model                                        # Pass the prompt to the LLM for response generation
            | StrOutputParser()                                     # Parse the string output from the LLM
        )
        return rag_chain
    


######## Hybrid Search / Ensemble Retriever ########    
    
class EnsembleRetrieving:
    def __init__(self, chroma_path, embedding_function, llm_model, prompt_template, k:int=3, chroma_path_2="bm25",chroma_path_3 = None, reranker=False):
        self.chroma_path = chroma_path
        self.chroma_path_2 = chroma_path_2
        self.chroma_path_3 = chroma_path_3
        
        try:
            self.embedding_model = embedding_function
        except Exception as e:           
            logging.error(f"Error initializing embedding model: {e}")
            exit(1)
        try:
            self.llm_model = llm_model     
        except Exception as e:
            logging.error(f"Error initializing llm model {e}")    
            exit(1)
        self.reranker = reranker
            
        self.db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_model)
        self.retriever_1= self.db.as_retriever(search_kwargs={"k": k})
        
        if self.chroma_path_2 == "bm25":
            with open('chroma_db/BM25_retriever/bm25_retriever.pkl', 'rb') as f:
                self.retriever_2 = pickle.load(f)
            self.retriever_2.k = k
        else:
            self.retriever_2 = Chroma(persist_directory=self.chroma_path_2, embedding_function=self.embedding_model).as_retriever(search_kwargs={"k": k})
        
        if self.chroma_path_3:
            self.retriever_3 = Chroma(persist_directory=self.chroma_path_3, embedding_function=self.embedding_model).as_retriever(search_kwargs={"k": k})
        else:
            self.retriever_3 = None
            
        if self.reranker:
            retrievers = [self.retriever_1, self.retriever_2]
            if self.retriever_3:
                retrievers.append(self.retriever_3)
            self.retriever = EnsembleRetriever(retrievers=retrievers, weights=[])
            self.model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
            self.compressor = CrossEncoderReranker(model=self.model, top_n=3)
            self.ensemble_retriever = ContextualCompressionRetriever(base_compressor=self.compressor, base_retriever=self.retriever)
        else:
            retrievers = [self.retriever_1, self.retriever_2]
            if self.retriever_3:
                retrievers.append(self.retriever_3)
            self.ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=[])
            
        self.prompt_template = ChatPromptTemplate.from_template(prompt_template)
    
    def format_docs(self, results):
        return  "\n\n---\n\n".join([doc.page_content for doc in results])
    
    def rag_chain(self):
        rag_chain = (
            {
                "context": self.ensemble_retriever | self.format_docs,       # Retrieve top 3 docs and format them
                "question": RunnablePassthrough()                   # Pass the question through without modification
            }
            | self.prompt_template                                  # Format the prompt with context and question
            | self.llm_model                                        # Pass the prompt to the LLM for response generation
            | StrOutputParser()                                     # Parse the string output from the LLM
        )
        return rag_chain

    


######## Reranker ########

class Reranker:
    def __init__(self, chroma_path, embedding_function, llm_model, prompt_template, retriever="naive", k=10):
        """
        Args:
        retriever: str - Type of retriever to use. Options are "naive", "multi_query", "parent_document_retriever", "pdr_multi_query"
        """
        
        if retriever not in ["naive", "multi_query", "parent_document_retriever", "pdr_multi_query"]:
            print("Invalid retriever type")
            exit(1)
        
        self.chroma_path = chroma_path
        try:
            self.embedding_model = embedding_function
        except Exception as e:           
            logging.error(f"Error initializing embedding model: {e}")
            exit(1)
        try:
            self.llm_model = llm_model     
        except Exception as e:
            logging.error(f"Error initializing llm model: {e}")    
            exit(1)
        
        self.db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_model)
        self.k = k
        self.retriever = retriever
        
        if self.retriever == "parent_document_retriever":
            self.retriever = ParentDocumentRetriever(client=self.db, k=self.k)
        elif self.retriever == "multi_query":
            self.retriever = MultiQueryRetriever.from_llm(retriever=self.db.as_retriever(search_kwargs={"k": self.k}), llm=self.llm_model)
        elif self.retriever == "pdr_multi_query":
            self.retriever = MultiQueryRetriever.from_llm(retriever=ParentDocumentRetriever(client=self.db, k=self.k), llm=self.llm_model)
        else:
            self.retriever = self.db.as_retriever(search_kwargs={"k": self.k})
        
        self.model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")  # Reranker model
        self.compressor = CrossEncoderReranker(model=self.model)
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=self.compressor, base_retriever=self.retriever)
        
        self.prompt_template = ChatPromptTemplate.from_template(prompt_template)
        
    def format_docs(self, results):
        return  "\n\n---\n\n".join([doc.page_content for doc in results])

    def rag_chain(self):
        rag_chain = (
            {
                "context": self.compression_retriever | self.format_docs,       # Retrieve top 3 docs and format them
                "question": RunnablePassthrough()                   # Pass the question through without modification
            }
            | self.prompt_template                                  # Format the prompt with context and question
            | self.llm_model                                        # Pass the prompt to the LLM for response generation
            | StrOutputParser()                                     # Parse the string output from the LLM
        )
        return rag_chain


##### Hybrid Search (Ensemble) / Recursive + Parent Document Retriever + (Semantic) #####

class EnsembleParentDocumentRetriever:  
    """
    args:
    chroma_path: str - Path to the chroma database
    chroma_path_pdr: str - Path to the parent document retriever database
    """
    
    def __init__(self, chroma_path, chroma_path_pdr, embedding_function, llm_model, prompt_template, k:int=3, window_size:int=2,chroma_semantic=None,reranker=False, multi_query=False):    
        self.chroma_path = chroma_path
        self.chroma_path_2 = chroma_path_pdr
        self.chroma_semantic = chroma_semantic
        try:
            self.embedding_model = embedding_function
        except Exception as e:           
            logging.error(f"Error initializing embedding model: {e}")
            exit(1)
        try:
            self.llm_model = llm_model     
        except Exception as e:
            logging.error(f"Error initializing llm model: {e}")    
            exit(1)
        self.reranker = reranker
        self.multi_query = multi_query
            
        self.db = Chroma(persist_directory=self.chroma_path, embedding_function=self.embedding_model)
        if self.multi_query:
            self.retriever_1 = MultiQueryRetriever.from_llm(retriever=self.db.as_retriever(search_kwargs={"k": k}), llm=self.llm_model)
        else:
            self.retriever_1= self.db.as_retriever(search_kwargs={"k": k})
        
        self.db2 = Chroma(persist_directory=self.chroma_path_2, embedding_function=self.embedding_model)
        if self.multi_query:
            self.retriever_2 = MultiQueryRetriever.from_llm(retriever=ParentDocumentRetriever(client=self.db2, window_size=window_size, k=k), llm=self.llm_model)
        else:
            self.retriever_2 =  ParentDocumentRetriever(client=self.db2, window_size=window_size, k=k)
        
        if self.chroma_semantic:
            self.db3 = Chroma(persist_directory=self.chroma_semantic, embedding_function=self.embedding_model)
            if self.multi_query:
                self.retriever_3 = MultiQueryRetriever.from_llm(retriever=self.db3.as_retriever(search_kwargs={"k": k}), llm=self.llm_model)
            else:
                self.retriever_3 = self.db3.as_retriever(search_kwargs={"k": k})
        
        if self.reranker:
            retrievers = [self.retriever_1, self.retriever_2]
            if self.chroma_semantic:
                retrievers.append(self.retriever_3)
            self.retriever = EnsembleRetriever(retrievers=retrievers, weights=[])
            self.model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
            self.compressor = CrossEncoderReranker(model=self.model, top_n=3)
            self.ensemble_retriever = ContextualCompressionRetriever(base_compressor=self.compressor, base_retriever=self.retriever)
        
        else:
            retrievers = [self.retriever_1, self.retriever_2]
            if self.chroma_semantic:
                retrievers.append(self.retriever_3)
            self.ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=[])
        
        self.prompt_template = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(self, results):
        return  "\n\n---\n\n".join([doc.page_content for doc in results])
    def rag_chain(self):
        rag_chain = (
            {
                "context": self.ensemble_retriever | self.format_docs,       # Retrieve top 3 docs and format them
                "question": RunnablePassthrough()                   # Pass the question through without modification
            }
            | self.prompt_template                                  # Format the prompt with context and question
            | self.llm_model                                        # Pass the prompt to the LLM for response generation
            | StrOutputParser()                                     # Parse the string output from the LLM
        )
        return rag_chain