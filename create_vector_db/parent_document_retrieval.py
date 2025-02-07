from langchain_community.document_loaders import UnstructuredXMLLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import google.generativeai as genai
import shutil


load_dotenv()


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])



CHROMA_PATH = r"chroma_db\chroma_parent_retriever_512"  # Path to the directory where the database will be saved and name of the database
DATA_PATH = r"create_vector_db\data"                                       # Path to the directory containing the knowledge base documents

embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") # Embedding model

def preprocess_file(file_path: str) -> list[Document]:
    """Load pdf file, chunk and build appropriate metadata"""
    loader = DirectoryLoader(DATA_PATH, glob="./*.xml", loader_cls=UnstructuredXMLLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=0,  #No overlap between chunks because we want to retrieve surrounding chunks later
    )

    docs = text_splitter.split_documents(documents=documents)
    chunks_metadata = [
        {"document_id": file_path, "sequence_number": i} for i, _ in enumerate(docs)  
    ]
    for chunk, metadata in zip(docs, chunks_metadata):
        chunk.metadata.update(metadata)

        
    return docs


def get_chroma() -> Chroma:
    return Chroma(embedding_function=embedding, persist_directory=CHROMA_PATH)


def main():
    documents = preprocess_file(file_path=DATA_PATH)

    
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Chroma example
    chroma = get_chroma()
    
    batch_size = 166  # Number of documents to process at a time, this can change depending on your system. See Chromadb documentation for more details.
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        chroma.add_documents(documents=batch)
        

    print(f"Saved {len(documents)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
