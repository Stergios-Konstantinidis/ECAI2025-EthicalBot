import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import csv
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import openai
from langchain.retrievers import EnsembleRetriever
from rag_models import ParentDocumentRetriever
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError as e:
    print(f"Environment variable not found: {e}")
    exit(1)

CHROMA_PATH = r"chroma_db\chroma_recursive_2048"
CHROMA_PATH_PDR = r"chroma_db\chroma_parent_retriever_512"


PROMPT_TEMPLATE = """
Answer the question based only on the following context which is delimited with triple backticks:

'''{context}'''

---

If the question cannot be answered based on the context, start your answer with "Cannot answer: " followed by the reason. Otherwise, provide a detailed answer.

Question: {question}
"""

def main(query:str, max_attempts:int = 4):

    
    original_query = query

    try:
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    except Exception as e:
        print(f"Error initializing OpenAIEmbeddings: {e}")
        exit(1)
        
    db1 = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    db2 = Chroma(persist_directory=CHROMA_PATH_PDR, embedding_function=embedding_function)
    
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    
    retriever_1 = MultiQueryRetriever.from_llm(retriever=db1.as_retriever(search_kwargs={"k": 1}), llm=model, parser_key="lines")
    retriever_2 = MultiQueryRetriever.from_llm(retriever=ParentDocumentRetriever(db2, k=1), llm=model, parser_key="lines")
  
    ensemble_retriever = EnsembleRetriever(retrievers=[retriever_1, retriever_2], weights=[])

    current_query = original_query
    used_queries = []
    
    for attempt in range(max_attempts):
        # Retrieve relevant documents
        results = ensemble_retriever.invoke(current_query)

        used_queries.append(f"Attempt {attempt+1}: {current_query}")
        context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
        # Generate answer
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=current_query)
        
        response = model.invoke(prompt)
        
        response_text = response.content.strip()
        
        # Check if answerable
        if not response_text.startswith("Cannot answer: "):
            print(response_text)
            return
        else:
            if attempt < max_attempts - 1:
                # Rephrase query
                reason = response_text[len("Cannot answer: "):]
                rephrase_prompt = f"""The original question '{current_query}' couldn't be answered because: {reason}.
The knowledge base used to answer the question is an old text, rephrase this question into a more general form using common terms that could be understood in the context of an old text. Avoid modern concepts and technology. 
This is your attempt {attempt+1}. With each attempt, the question should lose specific details and become more abstract, ultimately leading to a broad ethical question. Return only the rephrased query.

Examples:

1) used_queries = ["Attempt 1: Is it ethical to use artificial intelligence to replace human workers?"], Original: "Is it ethical to use artificial intelligence to replace human workers?", Rephrased: "Is it right to replace human work with the labor of tools or mechanical devices?"
   used_queries = ["Attempt 1: Is it ethical to use artificial intelligence to replace human workers?", "Attempt 2: Is it right to replace human work with the labor of tools or mechanical devices?"], Original: "Is it right to replace human work with the labor of tools or mechanical devices?", Rephrased: "Is it fair to replace the work of people with machines?"  
   used_queries = ["Attempt 1: Is it ethical to use artificial intelligence to replace human workers?", "Attempt 2: Is it right to replace human work with the labor of tools or mechanical devices?", "Attempt 3: Is it fair to replace the work of people with machines?"], Original: "Is it fair to replace the work of people with machines?", Rephrased: "Should we prefer tools or people for work?"
 

2) used_queries = ["Attempt 1: Who should be responsible if an AI that I own causes harm?"], Original: "Who should be responsible if an AI that I own causes harm?", Rephrased: "Who is to blame if a machine that I own causes harm?"
    used_queries = ["Attempt 1: Who should be responsible if an AI that I own causes harm?", "Attempt 2: Who is to blame if a machine that I own causes harm?"], Original: "Who is to blame if a machine that I own causes harm?", Rephrased: "Who is responsible if something that I own causes harm without my control?"
    used_queries = ["Attempt 1: Who should be responsible if an AI that I own causes harm?", "Attempt 2: Who is to blame if a machine that I own causes harm?", "Attempt 3: Who is responsible if something that I own causes harm without my control?"], Original: "Who is responsible if something that I own causes harm without my control?", Rephrased: "Who is responsible if I cause harm without meaning to?"


used_queries = {used_queries} Original: "{current_query}", Rephrased:"""
                
                
                rephrased = model.invoke([
                    SystemMessage(content="You rephrase questions to be simpler and more general, avoiding modern concepts and technology."),
                    HumanMessage(content=rephrase_prompt)
                ])
                current_query = rephrased.content.strip()
                print(f"Attempt {attempt+1} failed. Trying rephrased query: {current_query}")
            else:
                print(response_text)
                return

if __name__ == "__main__":
    #questions_dataset = "evaluate\dataset.csv"
    #with open(questions_dataset, newline='') as csvfile:
    #    reader = csv.DictReader(csvfile)
    #    dataset = [{"query": row['question']} for row in reader if row["difficulty"] == "nonanswerable"]
    #   
    #for row in dataset:
    #    main(row["query"])

    main("Should wealthy nations contribute more to combating climate change?", max_attempts=5)