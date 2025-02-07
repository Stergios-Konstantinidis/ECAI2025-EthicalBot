import shutil
from langchain_community.document_loaders import UnstructuredXMLLoader, DirectoryLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
import google.generativeai as genai


load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"]) 


CHROMA_PATH = r"chroma_db\chroma_recursive_2048" # Path to the directory where the database will be saved and name of the database
DATA_PATH = r"create_vector_db\data"                        # Path to the directory containing the knowledge base documents

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    
    loader = DirectoryLoader(DATA_PATH, glob="./*.xml", loader_cls=UnstructuredXMLLoader) # Select the loader according to the type of the documents
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=205,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # To see the content of a random chunk(30) and check if there is a problem with the text splitting
    document = chunks[30]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents. 
    batch_size = 166                                # Number of documents to process at a time, this can change depending on your system. See Chromadb documentation for more details.
    for i in range(0, len(chunks), batch_size):     
        batch = chunks[i:i + batch_size]
        db = Chroma.from_documents(
            batch, GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"), persist_directory=CHROMA_PATH
        )
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    

if __name__ == "__main__":
    main()