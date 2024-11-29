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

