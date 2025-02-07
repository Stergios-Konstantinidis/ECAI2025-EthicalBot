import openai
import os
from rouge_score import rouge_scorer
import erag
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_chroma import Chroma
from dotenv import load_dotenv
import csv
from rag_models import ParentDocumentRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
import pickle, time



load_dotenv()

try:
     openai.api_key = os.environ["OPENAI_API_KEY"]
except KeyError as e:
     print(f"Environment variable not found: {e}")
     exit(1)




PROMPT_TEMPLATE = """
You are a specialized assistant designed to answer ethical questions using insights from the Nicene and Post-Nicene Fathers, 
a landmark collection of early Christian writings. Your responses must draw **only and strictly** from the works of St. John Chrysostom, 
known for his practical teachings on social justice, morality, and Scripture.

Answer the question based solely on the provided context, which is delimited by triple backticks. If the context does not provide sufficient information to answer the question, clearly state that explicitly.

Context:
'''{context}'''

---

Provide an answer to the following question based solely on the provided context. If the context does not provide sufficient information to answer the question, state that explicitly.

Question:
{question}
"""


####This is the prompt template that was used for Flexible/Less restrictive Prompt evaluation.

#PROMPT_TEMPLATE = """
#You are a specialized assistant designed to answer ethical questions using insights from the Nicene and Post-Nicene Fathers, 
#a landmark collection of early Christian writings. Your responses should draw specifically from the works of St. John Chrysostom, 
#known for his practical teachings on social justice, morality, and Scripture.
#
#Answer the question based on the provided context, which is delimited by triple backticks. 
#
#Context:
#'''{context}'''
#
#---
#
#Provide an answer to the following question. If the context does not provide sufficient information to answer the question, state that explicitly.
#
#Question:
#{question}
#"""





#Paths to the chroma databases
CHROMA_PATH = r"chroma_db\chroma_recursive_2048"
CHROMA_PATH_SEMANTIC = r"chroma_db\chroma_semantic"
CHROMA_PATH_PDR = r"chroma_db\chroma_parent_retriever_512"



def search_documents(queries, num_results, retriever="naive",  reranker=False):
    """
    search_documents: This function retrieves documents from the database based on the queries provided.
    Args:
    queries: A list of queries to search for.
    num_results: The number of results to retrieve.
    retriever: The type of retriever to use. Options are "naive", "multi_query", "parent_document_retriever", "pdr_multi_query".
    reranker: A boolean indicating whether to use a reranker.
    
    """
    
    if retriever not in ["naive", "multi_query", "parent_document_retriever", "pdr_multi_query"]:
        print("Invalid retriever type")
        exit(1)
    
    results = dict() 
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    except Exception as e:
        print(f"Error initializing Embeddings: {e}")
        exit(1)
    
    
    
    if retriever == "parent_document_retriever" or retriever == "pdr_multi_query":
        db = Chroma(persist_directory=CHROMA_PATH_PDR, embedding_function=embedding_function)
    else:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
    if reranker:
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
        compressor = CrossEncoderReranker(model=model)
    for query in queries:
        if retriever == "multi_query":
            if reranker:
                retriever = MultiQueryRetriever.from_llm(retriever=db.as_retriever(search_kwargs={"k": num_results}), llm=ChatOpenAI(model="gpt-4o-mini",temperature=0))
                results_query = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever).invoke(query)
            else:
                results_query = MultiQueryRetriever.from_llm(retriever=db.as_retriever(search_kwargs={"k": num_results}), llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)).invoke(query) 
        elif retriever == "parent_document_retriever":
            if reranker:
                retriever = ParentDocumentRetriever(client=db, window_size=2, k=num_results)
                results_query = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever).invoke(query)
            else:
                results_query = ParentDocumentRetriever(client=db, window_size=2, k=num_results)._get_relevant_documents(query)
        elif retriever == "pdr_multi_query":
            if reranker:
                retriever = ParentDocumentRetriever(client=db, window_size=2, k=num_results)
                results_query = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=MultiQueryRetriever.from_llm(retriever=retriever, llm=ChatOpenAI(model="gpt-4o-mini",temperature=0))).invoke(query)
            else:
                results_query = MultiQueryRetriever.from_llm(retriever=ParentDocumentRetriever(client=db, window_size=2, k=num_results), llm=ChatOpenAI(model="gpt-4o-mini",temperature=0)).invoke(query)
        else:  
            if reranker:
                results_query = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=db.as_retriever(search_kwargs={"k": num_results})).invoke(query)
            else:
                results_query = db.similarity_search_with_relevance_scores(query, k=num_results)
                 
        docs =[]
        
        if retriever == "naive" and not reranker:
            for result, _score in results_query:
                docs.append(result.page_content)
        else:
            for result in results_query:
                docs.append(result.page_content)

        results[query] = docs
    return results

#Function to handle the quota limit error for embedding model
def invoke_with_retry(retriever, query, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return retriever.invoke(query)
        except GoogleGenerativeAIError as e:
            if attempt < retries - 1:
                print(f"Quota limit reached. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Augmentez le délai exponentiellement
            else:
                raise e


def search_documents_hybrid_search(queries, num_results,chroma1, reranker=False,chroma2="BM25", chroma3=None, multiquery=False):
    """
    search_documents_hybrid_search: Using ensemble retriever, this function uses multiple retrievers to retrieve documents from the database based on the queries provided.
    args:
    queries: A list of queries to search for.
    num_results: The number of results to retrieve.
    chroma1: The path to the first chroma database.
    reranker: A boolean indicating whether to use a reranker.
    chroma2: The path to the second chroma database. Use ["parent_document_retriever", "BM25", recursive or semantic chroma database path] Default is "BM25".
    chroma3: The path to the third chroma database. Default is None.
    
    """
    
    
    results = dict() 
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    except Exception as e:
        print(f"Error initializing Embeddings: {e}")
        exit(1)
    db1 = Chroma(persist_directory=chroma1, embedding_function=embedding_function)
    if multiquery:
        retriever1 = MultiQueryRetriever.from_llm(retriever=db1.as_retriever(search_kwargs={"k": num_results}), llm=ChatOpenAI(model="gpt-4o-mini",temperature=0))
    else:
        retriever1 = db1.as_retriever(search_kwargs={"k": num_results})
    
    if chroma2 == "BM25":
        with open('chroma_db/BM25_retriever/bm25_retriever.pkl', 'rb') as f:
            retriever_2 = pickle.load(f)
            retriever_2.k = num_results
    elif chroma2 == "parent_document_retriever":
        db2 = Chroma(persist_directory=CHROMA_PATH_PDR, embedding_function=embedding_function)
        if multiquery:
            retriever_2 = MultiQueryRetriever.from_llm(retriever=ParentDocumentRetriever(client=db2, k=num_results), llm=ChatOpenAI(model="gpt-4o-mini",temperature=0))
        else:
            retriever_2 = ParentDocumentRetriever(client=db2, window_size=2, k=num_results)
    else:
        retriever_2 = Chroma(persist_directory=chroma2, embedding_function=embedding_function).as_retriever(search_kwargs={"k": num_results})

    if chroma3:
        db3 = Chroma(persist_directory=chroma3, embedding_function=embedding_function)
        if multiquery:
            retriever3 = MultiQueryRetriever.from_llm(retriever=db3.as_retriever(search_kwargs={"k": num_results}), llm=ChatOpenAI(model="gpt-4o-mini",temperature=0))
        else:
            retriever3 = Chroma(persist_directory=chroma3, embedding_function=embedding_function).as_retriever(search_kwargs={"k": num_results})
    else:
        retriever3 = None
        
    for query in queries:
        if reranker:
            model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
            compressor = CrossEncoderReranker(model=model)
            retrievers =[retriever1, retriever_2]
            if retriever3:
                retrievers.append(retriever3)
            ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=[])
            results_query = invoke_with_retry(ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever), query, delay=5, retries=5)
        else:
            retrievers =[retriever1, retriever_2]
            if retriever3:
                retrievers.append(retriever3)
            results_query= invoke_with_retry(EnsembleRetriever(retrievers=retrievers, weights=[]), query, delay=5, retries=5)

        #print(len(results_query))
        docs=[]
        
        for result in results_query:
            docs.append(result.page_content)

        results[query] = docs
    return results

        
    

def text_generator(queries_and_documents):
 
        
    results = dict()
    for question, documents in queries_and_documents.items():
        context = ' '.join(documents)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        chat_completion = openai.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-4o-mini",
        )

        results[question] = chat_completion.choices[0].message.content

    return results


def rouge_metric(generated_outputs, expected_outputs):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    results = dict()
    for query, gen_output in generated_outputs.items():
        expe_outputs_query = expected_outputs[query]
        max_value = 0
        for exp_output in expe_outputs_query:
            max_value = max(scorer.score(exp_output, gen_output)['rougeL'].fmeasure, max_value)
        results[query] = max_value
    return results



def load_queries_and_expected_outputs(csv_file):
    queries = []
    expected_outputs = {}
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:     
            query = row["question"]
            expected_output = row["answer"]
            queries.append(query)
            expected_outputs[query] = [expected_output]  # Wrapping in a list to maintain compatibility
    return queries, expected_outputs

csv_file_path = r'evaluate\dataset.csv'
queries, expected_outputs = load_queries_and_expected_outputs(csv_file_path)


if __name__ == "__main__":
    
    retrieved_documents = search_documents(queries=queries, num_results=3, reranker=False, retriever="naive")
    #retrieved_documents = search_documents_hybrid_search(queries=queries, num_results=2, chroma1=CHROMA_PATH, reranker=True, chroma2="parent_document_retriever", multiquery=True)
    
    retrieval_metrics = {'P', 'success'}
    
    results = erag.eval(
    retrieval_results = retrieved_documents,
    expected_outputs = expected_outputs,
    text_generator = text_generator,
    downstream_metric = rouge_metric,
    retrieval_metrics = retrieval_metrics
    )
    
    print(results)
