"""Pinecone vector store management"""

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as lc_Pinecone


class PineconeStore:
    """
    Manages Pinecone vector database for document storage and retrieval.
    """

    def __init__(self, api_key, index_name='ckd-chatbot', dimension=384, verbose=True):
        """
        Initialize Pinecone store.

        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            dimension: Embedding dimension (default: 384 for all-MiniLM-L6-v2)
            verbose: Print progress information
        """
        self.api_key = api_key
        self.index_name = index_name
        self.dimension = dimension
        self.verbose = verbose
        self.pc = None
        self.index = None

    def init_index(self):
        """
        Initialize or connect to existing Pinecone index.

        Returns:
            Pinecone index object
        """
        if self.verbose:
            print(f"Initializing Pinecone index: {self.index_name}")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)

        # List all indexes
        active_indexes = self.pc.list_indexes().names()

        # Check if index exists
        if self.index_name not in active_indexes:
            if self.verbose:
                print(f"Creating new index: {self.index_name}")

            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

            if self.verbose:
                print(f"Index '{self.index_name}' created successfully")
        else:
            if self.verbose:
                print(f"Index '{self.index_name}' already exists, connecting...")

        # Get the index
        self.index = self.pc.Index(self.index_name)

        if self.verbose:
            print("Successfully connected to Pinecone index")

        return self.index

    def as_retriever(self, embeddings, k=3, search_type="similarity"):
        """
        Create a LangChain retriever for similarity search.

        Args:
            embeddings: HuggingFace embeddings object
            k: Number of documents to retrieve (default: 3)
            search_type: Type of search (default: "similarity")

        Returns:
            LangChain retriever object
        """
        if self.verbose:
            print(f"Creating retriever with k={k}, search_type={search_type}")

        # Connect to existing index with embeddings
        docsearch = lc_Pinecone.from_existing_index(
            index_name=self.index_name,
            embedding=embeddings
        )

        # Create retriever
        retriever = docsearch.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

        if self.verbose:
            print("Retriever created successfully")

        return retriever

    def populate_index(self, text_chunks, embeddings):
        """
        Populate the Pinecone index with document chunks.

        Args:
            text_chunks: List of document chunks from text_split()
            embeddings: HuggingFace embeddings object

        Returns:
            Number of documents added
        """
        if self.verbose:
            print(f"Populating index with {len(text_chunks)} text chunks...")

        # Embed and upsert documents
        docsearch = lc_Pinecone.from_documents(
            documents=text_chunks,
            index_name=self.index_name,
            embedding=embeddings
        )

        if self.verbose:
            print(f"Successfully populated index with {len(text_chunks)} chunks")

        return len(text_chunks)
