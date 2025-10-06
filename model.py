import os
from secret_manager import GEMINI_API_KEY

from langchain_community.document_loaders import WebBaseLoader # 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from google import genai
from google.genai.types import GenerateContentConfig

# --- Configuration ---
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
os.environ["USER_AGENT"] = "NewsResearchTool/1.0"

FOLDER_PATH = "faiss_index_local"
chat_model = "gemini-2.5-flash"
embedding_model = "text-embedding-004"

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

def process_urls(urls):
    print("🔄 Loading and processing URLs...")

    # 1. Load web content
    loader = WebBaseLoader(urls)
    data = loader.load()

    # 2. Split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(data)

    print("🔍 Creating FAISS vector store...")

    # 3. FAISS expects an object with .embed_documents() and .embed_query()
    #    So we’ll patch a simple object dynamically
    from types import SimpleNamespace
    embeddings = EmbeddingsWrapper()
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)


    # 4. Create FAISS index
    vectorstore.save_local(FOLDER_PATH)

    print("✅ FAISS index saved successfully at:", FOLDER_PATH)
    
def embed_documents(texts):
    """Embed a list of texts (used for document chunks)."""
    response = client.models.embed_content(
        model=embedding_model,
        contents=texts,
    )
    return [e.values for e in response.embeddings]


def embed_query(text):
    """Embed a single query (used during retrieval)."""
    response = client.models.embed_content(
        model=embedding_model,
        contents=text,
    )
    return response.embeddings[0].values

def chat(prompt, temperature=0.7):
    """Send a prompt to Gemini and return the response text."""
    response = client.models.generate_content(
        model=chat_model,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=2048,
        ),
    )
    return response.text

class EmbeddingsWrapper:
    def embed_documents(self, texts):
        return embed_documents(texts)
    def embed_query(self, text):
        return embed_query(text)
    def __call__(self, text):  # make callable for safety
        return embed_query(text)



def ask_question(question):
    print("💬 Asking:", question)

    # Same SimpleNamespace hack for FAISS compatibility
    from types import SimpleNamespace
    embeddings = EmbeddingsWrapper()

    # Load FAISS index
    vectorstore = FAISS.load_local(
        FOLDER_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    # Retrieve top relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)  # new API replacing get_relevant_documents()

    # Combine text context
    context = "\n\n".join(doc.page_content for doc in docs)
    sources = list({doc.metadata.get("source", "Unknown") for doc in docs})

    # Build and send prompt
    prompt = f"""Answer the following question based on the context below:

Context:
{context}

Question:
{question}

Answer:"""

    answer = chat(prompt)
    print("\n🧠 Gemini Answer:\n", answer)
    print("\n📚 Sources:")
    for src in sources:
        print("-", src)

#======================================================= RUN =================================================
print("**************** NEWS RESEARCH TOOL ************")
print("Press 1 to load url.")
print("Press 2 to ask questions.")
print("Press 3 to exit.")
urls = []
while(1):
    choice = str(input("Choice: "))
    

    if choice == "1":
        url = str(input("Give url: "))
        urls.append(url)
        process_urls(urls)
        
    elif choice == "2":
        if not urls:
            print("PLEASE GIVE VALID URL!")
            continue
        
        question = str(input("Question: "))
        ask_question(question)

    elif choice == "3":
        print("********* TERMINATED ****************")
        break

    else:
        print("Invalid choice. Try again?")
        continue
    