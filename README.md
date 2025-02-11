# IJCAI2025-The Ethical Bot: Combining Retrieval-Augmented Generation Techniques with Theological Texts for Answering Ethical and Existential Questions

This repository contains the code and implementation details for the research paper titled *'The Ethical Bot: Combining Retrieval-Augmented Generation Techniques with Theological Texts for Answering Ethical and Existential Questions.'* The project explores the optimization of Retrieval-Augmented Generation (RAG) pipelines to improve user understanding of complex texts. It features a chatbot application designed to answer user questions, alongside the models, evaluation frameworks, and techniques discussed in the research. Detailed documentation is included to support replication and further exploration.

*This study centers on a knowledge base derived from theological works, particularly the writings of St. John Chrysostom from the Nicene and Post-Nicene Fathers.*



<div align="center">
  <img src="https://github.com/user-attachments/assets/222a6a75-b067-4df9-839c-d6166dba028b" alt="Image 1">
  <img src="https://github.com/user-attachments/assets/8a876fd7-eb88-45d8-8681-f8be72aef373" alt="Image 2">
</div>



---

## Repository Structure

The repository contains the following folders:

- **App**: Contains `backend` and `frontend` folders with `requirements.txt` and Dockerfiles to facilitate dockerizing the app.
- **chroma_db**: Includes pre-built vector store databases.
- **create_vector_db**: Contains Python files and data used for the creation of databases.
- **evaluate**: Includes evaluation frameworks, RAG models, and the dataset used in the study.

---

## How to Run the Chatbot App Locally

### Requirements

- Install the required Python libraries listed in the `requirements.txt` files.
- Fill the `.env` file (located either in the root folder or the backend folder, depending on how you run `app.py`) with your OpenAI and Gemini API keys.

### Steps

1. Run the backend `app.py`:
   - Ensure the correct path to your Chroma database is set in the backend app.
2. Run the frontend `app.py` using Streamlit:
   - Use the command: `streamlit run app.py`.

---

## How to Dockerize the Chatbot App

### Backend

1. Add your API keys to a `.env` file in the `backend` folder (include your OpenAI and Gemini API keys).
   - Note: While this is acceptable for testing purposes, it is not the most secure practice.

2. Dockerize the backend using the provided Dockerfile in the `backend` folder.

### Frontend

1. After the backend is online, copy the backend URL.
2. Replace the `FLASK_BACKEND_URL` variable in the frontend app with your backend URL.
3. Dockerize the frontend using the provided Dockerfile in the `frontend` folder.

---

## Evaluation

### Requirements

The required Python libraries are listed in the `requirements.txt` file inside the `evaluate` folder.

---

### Running the Evaluation

To replicate the evaluation, follow these steps:

1. **Set Up Your API Key**:  
   Add your API key to the `.env` file located in the root folder.  

2. **Using TruLens Evaluation**:  
   - Use the `evaluation_trulens.py` file to run evaluations with the TruLens framework.
   - All the different configurations are pre-defined but commented out for convenience, as running multiple models simultaneously can be time-consuming.
   - Uncomment the desired configurations to perform specific evaluations.

3. **Using eRAG Evaluation**:  
   - Use the `evaluation_eRAG.py` file to run evaluations with the eRAG framework.
   - Choose your desired model to evaluate by following the instructions in the `search_documents` and `search_documents_hybrid_search` methods.
   - Note: It is only possible to run one evaluation at a time with this file.

4. **Query Simplification**:

   To observe how the query simplification process iteratively modifies user queries, use the `query_simplification.py` file.

---

### Creating a New Database

If you wish to create a new vector database with different parameters:

1. Navigate to the `create_vector_db` folder.  
2. Use the provided Python files to configure and generate your database.  
3. The required Python libraries for this process are listed in the `requirements.txt` file within the `create_vector_db` folder.

---
