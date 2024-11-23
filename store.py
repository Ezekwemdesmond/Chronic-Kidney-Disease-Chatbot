import os
#from pinecone.grpc import PineconeGRPC as Pinecone
#import pinecone
from pinecone import  Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as lc_Pinecone
from dotenv import load_dotenv
from helper import load_pdf_file, text_split, download_hugging_face_embeddings, load_cached_embeddings


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data=load_pdf_file(data='data/')
text_chunks=text_split(extracted_data)
embeddings = load_cached_embeddings()


def init_pinecone_index(api_key, index_name, dimension=384):
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    
    # List all indexes
    active_indexes = pc.list_indexes().names()
    
    # Check if index exists
    if index_name not in active_indexes:
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    else:
        print(f"Index '{index_name}' already exists, connecting to it...")
    
    # Get the index
    index = pc.Index(index_name)
    return index


# Use the function
try:
    index_name = 'ckd-chatbot1'
    index = init_pinecone_index(
        api_key=PINECONE_API_KEY,
        index_name=index_name
    )
    print("Successfully connected to index")
    
except Exception as e:
    print(f"Error: {str(e)}")

# Embed each chunk and upsert the embeddings into your Pinecone index.
def populate_index(index, embeddings, text_chunks):
    docsearch = lc_Pinecone.from_documents(
        documents = text_chunks,
        index_name = index_name,
        embedding = embeddings,
    )

if __name__ == "__main__":
    populate_index(index, embeddings, text_chunks)
    print("Pinecone index populated successfully!")

'''

import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import pinecone
import hashlib

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = "ckd-chatbot"

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# Step 1: Load PDF documents
def load_pdf_files(data_dir):
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


# Step 2: Split documents into chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Step 3: Initialize HuggingFace Embeddings
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


# Step 4: Initialize Pinecone Index
def init_pinecone_index(api_key, index_name, dimension=384):
    # Initialize Pinecone
    pinecone.init(api_key=api_key, environment="us-east1-gcp")  # Adjust the environment as per your Pinecone setup

    # Create or connect to an existing index
    if index_name not in pinecone.list_indexes():
        print(f"Creating a new Pinecone index: {index_name}")
        pinecone.create_index(name=index_name, dimension=dimension, metric="cosine")
    else:
        print(f"Connecting to existing index: {index_name}")

    # Initialize and return the index
    index = pinecone.Index(index_name)
    return index


# Step 5: Generate a unique ID for each document
def generate_doc_id(content):
    # Hash the document content to generate a unique ID
    return hashlib.md5(content.encode("utf-8")).hexdigest()


# Step 6: Process and Upsert Only New Data
def process_and_upsert_new_data(documents, embeddings, index):
    # Prepare data for upsert
    new_data = []
    for doc in documents:
        doc_id = generate_doc_id(doc.page_content)
        vector = embeddings.embed_documents([doc.page_content])[0]
        new_data.append((doc_id, vector, {"id": doc_id}))

    # Batch upsert data
    if new_data:
        print(f"Uploading {len(new_data)} document embeddings to Pinecone...")
        index.upsert(vectors=new_data)
        print("Upload complete.")
    else:
        print("No new data to upload.")


# Main Workflow
if __name__ == "__main__":
    try:
        # Load and split documents
        documents = load_pdf_files(data_dir="data/")
        text_chunks = text_split(documents)
        print(f"Loaded and split {len(text_chunks)} document chunks.")

        # Download embeddings
        embeddings = download_hugging_face_embeddings()

        # Initialize Pinecone
        index = init_pinecone_index(api_key=PINECONE_API_KEY, index_name=INDEX_NAME)

        # Upsert data
        process_and_upsert_new_data(text_chunks, embeddings, index)
    except Exception as e:
        print(f"An error occurred: {e}")

'''






