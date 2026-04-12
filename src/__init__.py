"""
Chronic Kidney Disease Chatbot - Modular RAG Application

Modules:
    - ml_model: ML model training and prediction preprocessing
    - document_processing: PDF loading, text chunking, embeddings
    - vectorstore: Pinecone vector store management
    - hybridsearch: Hybrid BM25 + semantic retriever with RRF fusion
    - rag_pipeline: RAG pipeline and LLM integration
"""

from .ml_model import MLModelPipeline, preprocess_input
from .document_processing import load_documents, text_split, load_embeddings
from .vectorstore import PineconeStore
from .hybridsearch import HybridRetriever
from .rag_pipeline import RAGPipeline

__all__ = [
    "MLModelPipeline",
    "preprocess_input",
    "load_documents",
    "text_split",
    "load_embeddings",
    "PineconeStore",
    "HybridRetriever",
    "RAGPipeline",
]

__version__ = "2.0.0"
