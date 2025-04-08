import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# Load the GROQ API Key
groq_api_key = os.getenv('GROQ_API_KEY')

st.set_page_config(page_title="Kepler ASU Chatbot", page_icon="🤖")
st.image("logo.png", width=800)
st.markdown("### Chatbot Assistant", unsafe_allow_html=True)

# Sidebar info
st.sidebar.image("qr-code.png", width=200)
st.sidebar.markdown("<small>keplerasuscholars@asu.edu </small>", unsafe_allow_html=True)

# Initialize the LLM model with GROQ
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Chat prompt template to be used by the model
template = """
    You are a helpful assistant. 
    Answer the following questions considering the history of the conversation:
    Context from document: {context}
    Chat history: {chat_history}
    User question: {user_question}
"""

# Function to handle vector embeddings and document loading
def vector_embedding():
    if "vectors" not in st.session_state:
        # Use HuggingFace embeddings for document vectors
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Load documents from the "pdf" directory
        st.session_state.loader = PyPDFDirectoryLoader("./pdf")  # Data Ingestion
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        
        # Split documents into chunks of 1000 characters with 200 overlap
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        
        # Create FAISS vector store for efficient search
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vectorization
        # Set vectorstore in session state
        st.session_state.vectorstore = st.session_state.vectors

# Initialize vectorstore if not available
if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
    try:
        print("Processing PDF file now...")  # Log when processing starts
        with st.spinner("Processing document..."):
            vector_embedding()  # Create the vectorstore
        st.sidebar.success("ChatBot Initialized OK!")
    except Exception as e:
        st.sidebar.error(f"Error processing PDF: {str(e)}")
else:
    print("Vectorstore already exists, skipping PDF processing.")  # Log if vectorstore is already initialized

# Initialize chat history if not already initialized
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi, I'm a bot 🤖. How can I help you? Ask questions about this scholar's program, and I'll answer them!")
    ]

# Function to get relevant content from the PDF for the user's query
def get_relevant_content(user_query):
    if st.session_state.vectorstore is None:
        return "Sorry, I have not yet processed the PDF documents."
    
    # Perform similarity search in the vector store
    search_results = st.session_state.vectorstore.similarity_search(user_query, k=3)  # Search for top 3 relevant documents
    
    # Combine the content of the relevant documents into a string
    relevant_content = "\n".join([result.page_content for result in search_results])
    
    return relevant_content

# Display chat history with delete buttons and robot emoji for AI messages
for idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message("AI" if isinstance(message, AIMessage) else "Human"):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            if isinstance(message, AIMessage):
                st.write("🤖 " + message.content)  # Add robot emoji to AI messages
            else:
                st.write(message.content)
        with col2:
            if isinstance(message, HumanMessage):  # Only show delete button for user messages
                if st.button("🗑️", key=f"delete_{idx}", help="Delete this message"):
                    st.session_state.chat_history.pop(idx)  # Delete message from history

# Chat input
user_query = st.chat_input("Type your message here...")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
    
    with st.chat_message("AI"):
        # Get relevant content from the PDF based on the user's query
        relevant_content = get_relevant_content(user_query)
        
        # Trigger response from the model
        prompt = ChatPromptTemplate.from_template(template)
        chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())
        
        # Pass relevant context, conversation history, and user question to the prompt
        response = chain.run({
            "context": relevant_content,  # Pass the relevant document context
            "chat_history": st.session_state.chat_history,
            "user_question": user_query
        })
        
        # Display the AI's response with a robot emoji 🤖
        st.write("🤖 " + response)

        # Add the AI response to the chat history
        st.session_state.chat_history.append(AIMessage(content=response))
