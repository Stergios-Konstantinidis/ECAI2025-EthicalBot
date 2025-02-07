from flask import Flask, request, jsonify, Response
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import re
from langchain_openai import ChatOpenAI
import openai
from langchain_core.prompts import MessagesPlaceholder
 


load_dotenv()

# Configure Google API
try:
    openai.api_key = os.environ["OPENAI_API_KEY"]
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError as e:
    print(f"Environment variable not found: {e}")
    exit(1)
    

CHROMA_PATH = r"chroma"  # !!!! Path to the Chroma DB directory, set it to the correct path



PROMPT_TEMPLATE = """
You are a specialized assistant designed to answer ethical questions using insights from the Nicene and Post-Nicene Fathers, 
a landmark collection of early Christian writings. Your responses should draw specifically from the works of St. John Chrysostom, 
known for his practical teachings on social justice, morality, and Scripture.

Answer the question based solely on the provided context, which is delimited by triple backticks. 

Context:
'''{context}'''

---

Provide an answer to the following question. Avoid using phrases like "according to the provided context". If the context does not provide sufficient information to answer the question, state that explicitly.

Question:
{question}
"""

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# Initialize Flask app
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate_response():
    # Retrieve query_text from the POST request's JSON body
    data = request.json
    
    if data is None:
        return jsonify({"error": "No JSON body provided"}), 400
    
    query_text = data.get('query_text') # The user's question
    history = data.get('history', [])   # The chat history
    
    if not query_text:
        return jsonify({"error": "query_text is required"}), 400

    # Check and alidate history format
    if not all(isinstance(msg, (list, tuple)) and len(msg) == 2 for msg in history):
        return jsonify({"error": "Invalid history format"}), 400

    # Initialize embedding model
    try:
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") 
    except Exception as e:
        return jsonify({"error": f"Error initializing Embeddings: {e}"}), 500

    # Initialize Chroma DB
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Combine history and current query for context
    combined_context = [{"role": "user", "content": msg[0]} if i % 2 == 0 else {"role": "assistant", "content": msg[1]} for i, msg in enumerate(history)]
    combined_context.append({"role": "user", "content": query_text})

    # Create a standalone question using the LLM with ChatPromptTemplate
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    reformulate_prompt = contextualize_q_prompt.format(chat_history=combined_context, input=query_text)
   
    # Generate new question based on the chat history and the user's question
    try:
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        standalone_question = model.predict(reformulate_prompt)
        
    except Exception as e:
        return jsonify({"error": f"Error generating standalone question: {e}"}), 500

    # Search the database using the new question
    try:
        results = db.similarity_search_with_relevance_scores(standalone_question, k=3) # Naive RAG
    
    except Exception as e:
        return jsonify({"error": f"Error searching Chroma DB: {e}"}), 500

    if not results:
        return jsonify({"error": "No matching results found"}), 404

    # Create the context from the results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    

    # Collect sources
    sources_set = set()
    sources = []
    for doc, _score in results:
        source_path = doc.metadata.get("source", "")
        match = re.search(r'npnf\d+', source_path)
        if match:
            source_name = match.group(0)
            if source_name not in sources_set:
                sources_set.add(source_name)
                url = generate_url(source_name) # Generate URL for the source
                link = f'[{source_name}]({url})' 
                sources.append(link)

    sources_text = "\n\nSource(s): " + ", ".join(sources) if sources else "\n\nSources: None"

    # Stream the response
    def generate_chunks():
        """Stream chunks of the response."""
        try:
            response_model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
            
            # Send the reponse first chunk by chunk
            for chunk in response_model.stream(prompt):
                yield chunk.content

            # Send sources at the end
            yield f"{sources_text}\n\n"
            
        except Exception as e:
            yield f"data: Error generating response: {str(e)}\n\n"
    
    # Return a stream of chunks
    return Response(generate_chunks(), content_type="text/event-stream")

# Generate URL for the source based on the source name
def generate_url(source_name):
    base_url = "https://www.ccel.org/ccel/schaff/"
    return f"{base_url}{source_name}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, threaded=True)