import streamlit as st
import requests

# Set the URL of the Flask backend
FLASK_BACKEND_URL = "http://localhost:8080/generate" 


def query_streaming_chatbot(query_text, history):
    """
    Sends a query to the chatbot backend with streaming enabled.
    """
    payload = {"query_text": query_text, "history": history}
    try:
        response = requests.post(FLASK_BACKEND_URL, json=payload, stream=True)
        if response.status_code == 200:
            
            # Stream the response chunk by chunk
            for chunk in response.iter_content(decode_unicode=True):
                yield chunk
                
        else:
            yield f"Error: Received status code {response.status_code}"
    except Exception as e:
        yield f"Error: Could not retrieve the response. Exception: {str(e)}"

def hide_buttons():
    st.markdown(
        """
        <style>
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="stBaseButton-primary"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Initialize the backend by sending a warm-up request, it's useful to avoid cold starts on cloud platforms when the server is not always running 
def warm_up_backend():
    """
    Sends a warm-up request to the backend to ensure it is ready.
    """
    payload = {"query_text": "warm-up", "history": []}
    try:
        response = requests.post(FLASK_BACKEND_URL, json=payload)
        if response.status_code == 200:
            print("Backend warmed up successfully.")
        else:
            print(f"Backend warm-up failed with status code {response.status_code}.")
    except Exception as e:
        print(f"Backend warm-up failed with exception: {str(e)}")


### Streamlit App

# Warm up the backend when the app is launched
if "backend_warmed_up" not in st.session_state:
    warm_up_backend()
    st.session_state.backend_warmed_up = True


st.set_page_config(page_title="Ethical Bot", layout="centered")
st.title("The Ethical Bot")


# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []  # Each message is a dict with 'user' and 'bot' keys
    
# Session state for the current page 
if "page" not in st.session_state:
    st.session_state.page = "home"

if "suggestion" not in st.session_state:
    st.session_state.suggestion = None
    
# Function to switch to the chat page with a suggestion
def switch_to_chat(suggestion):
    hide_buttons()
    st.session_state.page = "chat"
    st.session_state.history.append({"user": suggestion, "bot": None})
    st.rerun()

# Home page
if st.session_state.page == "home":
    
    
    st.markdown("Welcome to The Etchical Bot, this chatbot is designed to help answer ethical questions using insights "
                "from the Nicene and Post-Nicene Fathers, a landmark collection of early Christian writings. "
                "It draws specifically from the works of St. John Chrysostom, known for "
                "his practical teachings on social justice, morality, and Scripture.")
    st.markdown("### Try to ask a question:")
    
    
    
    if st.button("What are the benefits of being kind to strangers?", use_container_width=True):
        switch_to_chat("What are the benefits of being kind to strangers?")
    if st.button("How can I be more open-minded toward new ideas?", use_container_width=True):
        switch_to_chat("How can I be more open-minded toward new ideas?")
    if st.button("How can I become a better person?", use_container_width=True):
        switch_to_chat("How can I become a better person?")
    if st.button("Chat with The Ethical Bot",icon=":material/chat:", type="primary", use_container_width=True):
        st.session_state.page = "chat"
        st.rerun()


# Chat page
if st.session_state.page == "chat":
    # Chat interface using st.chat_message
    for message in st.session_state.history:
        with st.chat_message("user"):
            st.write(message["user"])
        if message["bot"]:
            with st.chat_message("assistant"):
                st.markdown(message["bot"])

    # User input
    if user_input := st.chat_input("Type your question here..."):
        # Append the user's input to the chat history
        st.session_state.history.append({"user": user_input, "bot": None})

        # Keep only the last 5 messages in the history
        if len(st.session_state.history) > 5:
            st.session_state.history = st.session_state.history[-5:]

        st.rerun()

    # Query the backend and update the chat history
    if st.session_state.history and st.session_state.history[-1]["bot"] is None:
        user_message = st.session_state.history[-1]["user"]
        history = [(msg["user"], msg["bot"]) for msg in st.session_state.history if msg["bot"]]

        with st.chat_message("assistant"):
            response = st.write_stream(query_streaming_chatbot(user_message, history))
           
            
        st.session_state.history[-1]["bot"] = response
        st.rerun()
