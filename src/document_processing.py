"""Document loading and embedding generation"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import joblib


def load_documents(data_dir='data/', verbose=True):
    """
    Load all PDF files from the data directory.

    Args:
        data_dir: Path to directory containing PDF files
        verbose: Print progress information

    Returns:
        List of LangChain document objects
    """
    if verbose:
        print(f"Loading PDF documents from: {data_dir}")

    loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()

    if verbose:
        print(f"Loaded {len(documents)} documents from PDFs")

    return documents


def text_split(extracted_data, chunk_size=500, chunk_overlap=20):
    """
    Split documents into smaller chunks for embedding.

    Args:
        extracted_data: List of documents from load_documents()
        chunk_size: Maximum size of each text chunk (default: 500)
        chunk_overlap: Number of overlapping characters (default: 20)

    Returns:
        List of chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks


def download_hugging_face_embeddings():
    """
    Download HuggingFace embeddings model.

    Returns:
        HuggingFaceEmbeddings: Embedding model (384 dimensions)
    """
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    return embeddings


def load_embeddings(cache_path="embeddings.pkl", verbose=True):
    """
    Load cached embeddings or download if not found.

    Args:
        cache_path: Path to cached embeddings file
        verbose: Print progress information

    Returns:
        HuggingFaceEmbeddings: Embedding model
    """
    try:
        if verbose:
            print("Loading cached embeddings...")
        embeddings = joblib.load(cache_path)
        if verbose:
            print("Embeddings loaded from cache.")
        return embeddings
    except FileNotFoundError:
        if verbose:
            print("Cache not found. Downloading embeddings...")
        embeddings = download_hugging_face_embeddings()
        joblib.dump(embeddings, cache_path)
        if verbose:
            print(f"Embeddings downloaded and cached to {cache_path}")
        return embeddings
