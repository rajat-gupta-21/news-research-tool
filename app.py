# app.py - Piece 1
import os
import streamlit as st
from secret_manager import GEMINI_API_KEY # llm api key
# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# --- Basic Setup ---
# Load environment variables from your .env file
# load_dotenv()
os.environ['GOOGLE_API_KEY'] = GEMINI_API_KEY

# Configure the Streamlit page
st.set_page_config(page_title="Chat With News Articles", layout="wide")
st.title("📰 Chat With News Articles: Gemini Edition")

# Path for the local FAISS vector store
FOLDER_PATH = "faiss_index_streamlit"

# app.py - Piece 2

# --- Model and Embeddings Initialization ---
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
except Exception as e:
    st.error(f"Error initializing Google AI models. Please check your API key. Error: {e}")
    st.stop()

# --- Session State Initialization ---
# This is to store the chat history and other state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed" not in st.session_state:
    st.session_state.processed = os.path.isdir(FOLDER_PATH)

# app.py - Piece 3

# --- Sidebar for URL Input and Processing ---
with st.sidebar:
    st.header("Process URLs")
    urls_input = st.text_area("Enter news article URLs (one per line):", height=150)
    
    if st.button("Process"):
        if not urls_input:
            st.warning("Please enter at least one URL.")
        else:
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            with st.spinner("Processing URLs... This might take a moment ⏳"):
                try:
                    # 1. Load data from URLs
                    loader = WebBaseLoader(urls)
                    data = loader.load()

                    # 2. Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                    docs = text_splitter.split_documents(data)

                    # 3. Create FAISS vector store and save it
                    vectorstore = FAISS.from_documents(docs, embeddings)
                    vectorstore.save_local(FOLDER_PATH)
                    
                    # 4. Update session state
                    st.session_state.processed = True
                    st.success("URLs processed successfully! ✅")
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")

# app.py - Piece 4

# --- Main Chat Interface ---

# Display previous chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask a question about the provided articles..."):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if URLs have been processed before chatting
    if not st.session_state.processed:
        st.warning("Please process URLs in the sidebar before asking questions.")
    else:
        with st.spinner("Thinking... 🤔"):
            # Load the vector store
            vectorstore = FAISS.load_local(
                FOLDER_PATH, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            
            # Create a stateless chain (we manage history manually)
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )
            
            # Get previous messages for context
            chat_history = [
                (msg["content"], st.session_state.messages[i+1]["content"])
                for i, msg in enumerate(st.session_state.messages)
                if msg["role"] == "user" and i+1 < len(st.session_state.messages)
            ]

            # Invoke the chain with the new question and history
            result = qa_chain.invoke({
                "question": prompt,
                "chat_history": chat_history
            })
            
            # Format the response with sources
            response_text = result["answer"]
            if result.get("source_documents"):
                response_text += "\n\n**Sources:**\n"
                unique_sources = set(doc.metadata['source'] for doc in result["source_documents"])
                for source in unique_sources:
                    response_text += f"- {source}\n"

            # Add AI response to session state and display it
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            with st.chat_message("assistant"):
                st.markdown(response_text)

    