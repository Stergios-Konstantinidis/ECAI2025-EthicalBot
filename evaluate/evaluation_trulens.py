import csv
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from trulens.core import TruSession
from trulens.core import Feedback
import numpy as np
from trulens.apps.langchain import TruChain
from trulens.dashboard.display import get_feedback_result
from trulens.providers.openai import OpenAI
import openai
import trulens.dashboard as dashboard
from trulens.core import Select
from trulens.feedback import GroundTruthAgreement
from trulens.core.utils.threading import TP
from rag_models import NaiveRAG, ParentRetriver, MultiQuery, Reranker, EnsembleRetrieving, ParentDocumentRetrieverwithMultiQuery, EnsembleParentDocumentRetriever


load_dotenv()

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    openai.api_key = os.environ["OPENAI_API_KEY"]
    
except KeyError as e:
    print(f"Environment variable not found: {e}")
    exit(1)
    
TP.DEBUG_TIMEOUT = None # Set to None to disable timeout, otherwise evaluation will stop after 10 minutes  
session = TruSession()
session.reset_database()

#Paths to the chroma databases
CHROMA_PATH = r"chroma_db\chroma_recursive_2048"
CHROMA_PATH_SEMANTIC = r"chroma_db\chroma_semantic"
CHROMA_PATH_PDR = r"chroma_db\chroma_parent_retriever_512"




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



# Provider for feedbacks
provider = OpenAI(model_engine="gpt-4o-mini")

#dataset
questions_dataset = r"evaluate\dataset.csv"

with open(questions_dataset, newline='') as csvfile:
    reader = csv.DictReader(csvfile)     
    dataset= [{"query": row['question'], "expected_response": row['answer']} for row in reader]


#Metrics for Naive RAG, Parent Retriever, Multi Query
def metrics(chain_name, reranker):
    
    if reranker:
        context = Select.RecordCalls.first.steps__.context.first._get_relevant_documents.rets

    else:
        context = TruChain.select_context(chain_name)

    f_groundedness = (
        Feedback(
            provider.groundedness_measure_with_cot_reasons, name="Groundedness"
        )
        .on(context.collect())  # collect context chunks into a list
        .on_output()
    )

    
    f_answer_relevance = Feedback(
        provider.relevance_with_cot_reasons, name="Answer Relevance"
    ).on_input_output()

    
    f_context_relevance = (
        Feedback(
            provider.context_relevance_with_cot_reasons, name="Context Relevance"
        )
        .on_input()
        .on(context)
        .aggregate(np.mean) # type: ignore
    )
    
    f_groundtruth = Feedback(
            GroundTruthAgreement(dataset, provider=provider).agreement_measure,
        name="Ground Truth",
    ).on_input_output()

    return [f_groundedness, f_answer_relevance, f_context_relevance, f_groundtruth]


def metrics_ensemble(reranker, ensemble_size):

    if reranker:
        context_1 = Select.RecordCalls.first.steps__.context.first.base_retriever.retrievers[0]._get_relevant_documents.rets[:].page_content
        context_2 = Select.RecordCalls.first.steps__.context.first.base_retriever.retrievers[1]._get_relevant_documents.rets[:].page_content
        if ensemble_size == 3:
            context_3 = Select.RecordCalls.first.steps__.context.first.base_retriever.retrievers[2]._get_relevant_documents.rets[:].page_content
        ensemble_context = Select.RecordCalls.first.steps__.context.first.invoke.rets[:].page_content
        
    else:
        context_1 = Select.RecordCalls.first.steps__.context.first.retrievers[0]._get_relevant_documents.rets[:].page_content
        context_2 = Select.RecordCalls.first.steps__.context.first.retrievers[1]._get_relevant_documents.rets[:].page_content
        if ensemble_size == 3:
            context_3 = Select.RecordCalls.first.steps__.context.first.retrievers[2]._get_relevant_documents.rets[:].page_content
        ensemble_context = Select.RecordCalls.first.steps__.context.first.invoke.rets[:].page_content
    
    f_groundedness_2 = (
        Feedback(
            provider.groundedness_measure_with_cot_reasons, name="Groundedness_2"
        )
        .on(context_2.collect())  # collect context chunks into a list
        .on_output()
    )
    
    f_groundedness_1 = (
        Feedback(
            provider.groundedness_measure_with_cot_reasons, name="Groundedness_1"
        )
        .on(context_1.collect())  # collect context chunks into a list
        .on_output()
    )
    
    f_groundedness = (
        Feedback(
            provider.groundedness_measure_with_cot_reasons, name="Groundedness"
        )
        .on(ensemble_context.collect())  # collect context chunks into a list
        .on_output()
    )
    
    # Question/answer relevance between overall question and answer.
    f_answer_relevance = Feedback(
        provider.relevance_with_cot_reasons, name="Answer Relevance"
    ).on_input_output()
    
    # Context relevance between question and each context chunk.
    f_context_relevance_2 = (
        Feedback(
            provider.context_relevance_with_cot_reasons, name="Context Relevance_2"
        )
        .on_input()
        .on(context_2)
        .aggregate(np.mean)
    )
    
    f_context_relevance_1 = (
        Feedback(
            provider.context_relevance_with_cot_reasons, name="Context Relevance_1"
        )
        .on_input()
        .on(context_1)
        .aggregate(np.mean)
    )
    
    f_context_relevance = (
        Feedback(
            provider.context_relevance_with_cot_reasons, name="Context Relevance"
        )
        .on_input()
        .on(ensemble_context)
        .aggregate(np.mean)
    )
    
    f_groundtruth = Feedback(
            GroundTruthAgreement(dataset, provider=provider).agreement_measure,
        name="Ground Truth",
    ).on_input_output()
    
    if ensemble_size == 3:
        f_groundedness_3 = (
            Feedback(
                provider.groundedness_measure_with_cot_reasons, name="Groundedness_3"
            )
            .on(context_3.collect())  # collect context chunks into a list
            .on_output()
        )
        
        f_context_relevance_3 = (
            Feedback(
                provider.context_relevance_with_cot_reasons, name="Context Relevance_3"
            )
            .on_input()
            .on(context_3)
            .aggregate(np.mean)
        )
        
        #You can check the feedbacks for each retriever by commenting out the return statement below.
        #return [f_groundedness_3, f_groundedness_2, f_groundedness_1, f_groundedness, f_answer_relevance, f_context_relevance_3, f_context_relevance_2, f_context_relevance_1, f_context_relevance, f_groundtruth]
        
    #You can check the feedbacks for each retriever by commenting out the return statement below.
    #return [f_groundedness_2, f_groundedness_1, f_groundedness, f_answer_relevance, f_context_relevance_2, f_context_relevance_1, f_context_relevance, f_groundtruth]
    
    return [f_groundedness, f_answer_relevance, f_context_relevance, f_groundtruth]



def trulens_recorder(chain_name, app_id:str, reranker, ensemble, ensemble_size):

        return TruChain(
            chain_name,
            app_id=app_id,
            feedbacks= metrics_ensemble(reranker,ensemble_size) if ensemble else metrics(chain_name, reranker),
        )


def evaluate(rag_model, rag_name:str, reranker=False, ensemble=False, ensemble_size=2):
    counter = 0
    rag = rag_model.rag_chain()
    rag_recorder = trulens_recorder(rag, rag_name, ensemble=ensemble, reranker=reranker, ensemble_size=ensemble_size)

    for row in dataset:
        with rag_recorder as recording:
            rag.invoke(row["query"])
            counter += 1
            print(f"Counter: {counter}") #Counter to keep track of the number of questions answered
            
    
    last_record = recording.records[-1]
    get_feedback_result(last_record, "Groundedness")


######################## Evaluation of the models ############################

# Models can be evaluated by uncommenting the respective model and calling the evaluate function.
# However calling multiple models at once increases the time taken for evaluation. Therefore, it is recommended to evaluate one model at a time if dataset isn't really small.


## Naive RAG / Recursive ####
naive_rag = NaiveRAG(
        chroma_path=CHROMA_PATH, 
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
        llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
        prompt_template=PROMPT_TEMPLATE)

evaluate(naive_rag, "1. Naive RAG")

### Naive RAG / Semantic ####
naive_rag = NaiveRAG(
        chroma_path=CHROMA_PATH_SEMANTIC, 
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
        llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
        prompt_template=PROMPT_TEMPLATE)

evaluate(naive_rag, "2. Naive RAG - Semantic")

##### Parent Retriever ####
parent_retriever = ParentRetriver(
    chroma_path=CHROMA_PATH_PDR,
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
    llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
    prompt_template=PROMPT_TEMPLATE
)

evaluate(parent_retriever, "3. Parent Document Retriever")


#### Multi Query ####
#multiquery_retriver = MultiQuery(
#    chroma_path=CHROMA_PATH,
#    embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#    llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#    prompt_template=PROMPT_TEMPLATE
#)
#
#evaluate(multiquery_retriver, "4. Multi Query")

##
#### NaiveRag with Reranker #### 
#reranker_naive = Reranker(
#    chroma_path=CHROMA_PATH,
#    embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#    llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#    prompt_template=PROMPT_TEMPLATE,
#    k=6
#)
#
#evaluate(reranker_naive, "5. Naive Rag with Reranker, k=6", reranker=True)



#### Parent Retriever with Reranker ####
#parent_reranker = Reranker(
#    chroma_path=CHROMA_PATH_PDR,
#    embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#    llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#    prompt_template=PROMPT_TEMPLATE,
#    retriever="parent_document_retriever",
#    k=6
#)
#evaluate(parent_reranker, "6. Parent Document Retriever with Reranker, k=6", reranker=True)

####Multi Query with Reranker ####
#multiquery_reranker = Reranker(
#    chroma_path=CHROMA_PATH,
#    embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#    llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#    prompt_template=PROMPT_TEMPLATE,
#    retriever="multi_query",
#    k=6
#)
#
#evaluate(multiquery_reranker, "7. Multi Query with Reranker, k=6", reranker=True)

##### Hybrid Search / Recursive + BM25 ####

#ensemble_retriever = EnsembleRetrieving(
#        chroma_path=CHROMA_PATH, 
#        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#        llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#        prompt_template=PROMPT_TEMPLATE)
#
#evaluate(ensemble_retriever, "8. Hybrid Search / Recursive + BM25", ensemble=True)

#
#### Hybrid Search / Recursive + Semantic ####
#ensemble_retriever = EnsembleRetrieving(
#        chroma_path=CHROMA_PATH, 
#        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#        llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#        prompt_template=PROMPT_TEMPLATE,
#        chroma_path_2=CHROMA_PATH_SEMANTIC
#        )
#
#evaluate(ensemble_retriever, "9. Hybrid Search / Recursive + Semantic", ensemble=True)
    

#### Hybrid Search / Recursive + BM25 + Semantic ####
#ensemble_reranker = EnsembleRetrieving(
#        chroma_path=CHROMA_PATH, 
#        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#        llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#        prompt_template=PROMPT_TEMPLATE,
#        chroma_path_3=CHROMA_PATH_SEMANTIC,
#        )
#        
#
#evaluate(ensemble_reranker, "10. Hybrid Search / Recursive + BM25 + Semantic ", ensemble=True, ensemble_size=3)

##### Hybrid Search with reranker / Recursive + Bm25 ####

#ensemble_retriever = EnsembleRetrieving(
#        chroma_path=CHROMA_PATH, 
#        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#        llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#        prompt_template=PROMPT_TEMPLATE,
#        reranker=True,
#        k=5
#        )
#
#evaluate(ensemble_retriever, "11. Hybrid Search with Reranker / Recursive + BM25", ensemble=True, reranker=True)

####### Hybrid Search with reranker / Recursive + Semantic ####
#ensemble_retriever = EnsembleRetrieving(
#        chroma_path=CHROMA_PATH, 
#        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#        llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#        prompt_template=PROMPT_TEMPLATE,
#        chroma_path_2=CHROMA_PATH_SEMANTIC,
#        reranker=True,
#        k=5
#        )
#
#evaluate(ensemble_retriever, "12. Hybrid Search with Reranker / Recursive + Semantic", ensemble=True, reranker=True)

#### Hybrid Search with reranker / Recursive + BM25 + Semantic ####
#ensemble_reranker = EnsembleRetrieving(
#        chroma_path=CHROMA_PATH, 
#        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#        llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#        prompt_template=PROMPT_TEMPLATE,
#        chroma_path_3=CHROMA_PATH_SEMANTIC,
#        reranker=True,
#        k=3
#        )
#
#evaluate(ensemble_reranker, "13. Hybrid Search with Reranker / Recursive + BM25 + Semantic", ensemble=True, reranker=True, ensemble_size=3)


#### Parent Document Retriever with Multi Query ####
#pdr_multiquery = ParentDocumentRetrieverwithMultiQuery(
#    chroma_path=CHROMA_PATH_PDR,
#    embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#    llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#    prompt_template=PROMPT_TEMPLATE
#)
#    
#evaluate(pdr_multiquery, "14. Parent Document Retriever with Multi Query")

##### Parent Document Retriever with Multi Query - Reranker ####

#pdr_multiquery_reranker = Reranker(
#    chroma_path=CHROMA_PATH_PDR,
#    embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#    llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#    prompt_template=PROMPT_TEMPLATE,
#    retriever="pdr_multi_query",
#    k=6
#)
#
#evaluate(pdr_multiquery_reranker, "15. Parent Document Retriever with Multi Query - Reranker", reranker=True)


##### Hybrid Search / Recursuive + Parent Document Retriever ####

#ensemble_retriever = EnsembleParentDocumentRetriever(
#        chroma_path=CHROMA_PATH, 
#        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#        llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#        prompt_template=PROMPT_TEMPLATE,
#        chroma_path_pdr=CHROMA_PATH_PDR
#        )
#
#evaluate(ensemble_retriever, "16. Hybrid Search / Recursive + Parent Document Retriever ", ensemble=True)


##### Hybrid Search / Recursive + Parent Document Retriever + Semantic ####

#ensemble_pdr = EnsembleParentDocumentRetriever(
#        chroma_path=CHROMA_PATH, 
#        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#        llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#        prompt_template=PROMPT_TEMPLATE,
#        chroma_path_pdr=CHROMA_PATH_PDR,
#        chroma_semantic=CHROMA_PATH_SEMANTIC,
#        k=3
#        )
#
#evaluate(ensemble_pdr, "17. Hybrid Search / Recursive + Parent Document Retriever + Semantic", ensemble=True, ensemble_size=3)

##### Hybrid Search with reranking / Recursive + Parent Document Retriever + Semantic  ####

#ensemble_reranker = EnsembleParentDocumentRetriever(
#        chroma_path=CHROMA_PATH, 
#        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#        llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#        prompt_template=PROMPT_TEMPLATE,
#        chroma_path_pdr=CHROMA_PATH_PDR,
#        chroma_semantic=CHROMA_PATH_SEMANTIC,
#        reranker=True,
#        k=3
#        )
#
#evaluate(ensemble_reranker, "18. Hybrid Search Reranker/ Recursive + Parent Document Retriever + Semantic", ensemble=True, reranker=True, ensemble_size=3)

#### Multi Query with Parent Document Retriever + Recursive ####

#multiquery_pdr_recursive = EnsembleParentDocumentRetriever(
#    chroma_path=CHROMA_PATH,
#    embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#    llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#    prompt_template=PROMPT_TEMPLATE,
#    chroma_path_pdr=CHROMA_PATH_PDR,
#    k=1,
#    multi_query=True
#)
#
#evaluate(multiquery_pdr_recursive, "19. Hybrid Search with Multi Query | Parent Document Retriever + Recursive", ensemble=True)


#### Multi Query with Parent Document Retriever + Recursive with Reranker ####

#ensemble_reranker = EnsembleParentDocumentRetriever(
#        chroma_path=CHROMA_PATH, 
#        embedding_function=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"),
#        llm_model=ChatOpenAI(model="gpt-4o-mini",temperature=0),
#        prompt_template=PROMPT_TEMPLATE,
#        chroma_path_pdr=CHROMA_PATH_PDR,
#        reranker=True,
#        k=2,
#        multi_query=True
#        )
#
#evaluate(ensemble_reranker, "20. Hybrid Search with Multi Query - Reranker| Parent Document Retriever + Recursive ", ensemble=True, reranker=True)


session.get_leaderboard()
dashboard.run_dashboard(port=59244)
