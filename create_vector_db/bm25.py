from langchain_community.document_loaders import UnstructuredXMLLoader, DirectoryLoader 
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever
from dotenv import load_dotenv
import pickle


load_dotenv()


DATA_PATH = r"create_vector_db\data" # Path to the directory containing the knowledge base documents 

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="./*.xml", loader_cls=UnstructuredXMLLoader)
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
    
    # To see the content of a random chunk and check if there is a problem with the text splitting
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):

    db = BM25Retriever.from_documents(chunks)
    
    with open('chroma_db/BM25_retriever/bm25_retriever.pkl', 'wb') as f:
        pickle.dump(db, f)

    
if __name__ == "__main__":
    main()
    