"""
Chronic Kidney Disease Chatbot - Modular RAG Application

Modules:
    - ml_model: ML model training and prediction preprocessing
    - document_processing: PDF loading, text chunking, embeddings
    - vectorstore: Pinecone vector store management
    - rag_pipeline: RAG chain and LLM integration
"""

from .ml_model import MLModelPipeline, preprocess_input
from .document_processing import load_documents, load_embeddings
from .vectorstore import PineconeStore
from .rag_pipeline import RAGPipeline

__all__ = [
    "MLModelPipeline",
    "preprocess_input",
    "load_documents",
    "load_embeddings",
    "PineconeStore",
    "RAGPipeline",
]

__version__ = "2.0.0"
