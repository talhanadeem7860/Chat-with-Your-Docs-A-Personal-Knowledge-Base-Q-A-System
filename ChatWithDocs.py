import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning) 

# --- 1. Document Loading and Processing ---
def create_vector_store_from_pdf(pdf_path):
    """
    Loads a PDF, splits it into chunks, creates embeddings,
    and stores them in a FAISS vector store.
    """
    
    db_file_path = f"{os.path.splitext(pdf_path)[0]}_faiss_index"
    if os.path.exists(db_file_path):
        print(f"[INFO] Loading existing vector store from {db_file_path}...")
        return FAISS.load_local(db_file_path, HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))

    print(f"[INFO] Creating new vector store for {pdf_path}...")
    # Load the document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    if not documents:
        print("[ERROR] Could not load any documents from the PDF.")
        return None

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    if not docs:
        print("[ERROR] Could not split the document into chunks.")
        return None

    # Create embeddings using a high-quality open-source model
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # Create the FAISS vector store and save it
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(db_file_path)
    print(f"[INFO] Vector store created and saved to {db_file_path}.")
    return db

# --- 2. Building the Q&A Chain ---
def create_qa_chain(db, llm):
    """
    Creates a retrieval-based Q&A chain.
    """
    # Define a prompt template to guide the LLM
    prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer from the context provided, just say that you don't know.
    Don't try to make up an answer.

    Context: {context}

    Question: {question}
    
    Helpful Answer:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create the Q&A chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "Stuff" all retrieved chunks into the prompt
        retriever=db.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 chunks
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- Main Application Logic ---
if __name__ == "__main__":
    # Check for OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("[ERROR] OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to run this application.")
    else:
        # Get PDF path from user
        pdf_path = input("Enter the full path to your PDF file: ").strip()
        
        if not os.path.exists(pdf_path):
            print(f"[ERROR] The file '{pdf_path}' does not exist.")
        else:
            # 1. Create or load the vector store
            vector_store = create_vector_store_from_pdf(pdf_path)
            
            if vector_store:
                # 2. Create the LLM and Q&A chain
                llm = OpenAI(temperature=0.1, openai_api_key=openai_api_key)
                qa_chain = create_qa_chain(vector_store, llm)
                
                # 3. Start interactive Q&A session
                print("\n[INFO] PDF processed. You can now ask questions.")
                print("Type 'exit' to quit.")
                
                while True:
                    user_question = input("\nYour Question: ")
                    if user_question.lower() == 'exit':
                        break
                    
                    # Get the answer from the chain
                    response = qa_chain({"query": user_question})
                    
                    # Print the answer
                    print("\nAnswer:")
                    print(response["result"])
                    
                    # Print the sources used
                    print("\nSources:")
                    for i, source in enumerate(response["source_documents"]):
                        print(f"  Source {i+1} (Page {source.metadata.get('page', 'N/A')}):")
                        print(f"    - \"{source.page_content[:150].strip()}...\"")