"""
Flask Web Application - Orchestrates all components
Chronic Kidney Disease Prediction and Management Chatbot
"""

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import time

from src import (
    MLModelPipeline,
    preprocess_input,
    load_embeddings,
    PineconeStore,
    RAGPipeline
)

# Load environment variables
load_dotenv()


class CKDChatbotCore:
    """
    Main application class orchestrating ML and RAG pipelines.
    Similar to KidneyCareAppCore pattern from KidneyCareAI.
    """

    def __init__(self, verbose=True):
        """
        Initialize the CKD Chatbot application.

        Args:
            verbose: Print initialization progress
        """
        self.verbose = verbose
        self.ml_pipeline = None
        self.rag_pipeline = None

        if self.verbose:
            print("\n" + "="*60)
            print("Initializing CKD Chatbot Core")
            print("="*60)

        self.initialize()

        if self.verbose:
            print("\nCKD Chatbot initialized successfully!")
            print("="*60 + "\n")

    def initialize(self):
        """Initialize all components: ML model and RAG pipeline."""
        # Initialize ML pipeline
        if self.verbose:
            print("\n1. Initializing ML Pipeline...")

        self.ml_pipeline = MLModelPipeline(verbose=self.verbose)
        self.ml_pipeline.load_model()

        # Initialize RAG pipeline
        if self.verbose:
            print("\n2. Initializing RAG Pipeline...")

        # Load embeddings
        embeddings = load_embeddings(verbose=self.verbose)

        # Initialize Pinecone vector store
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        vectorstore = PineconeStore(
            api_key=pinecone_api_key,
            index_name='ckd-chatbot',
            dimension=384,
            verbose=self.verbose
        )

        # Connect to existing index
        vectorstore.init_index()

        # Create retriever
        retriever = vectorstore.as_retriever(embeddings, k=3)

        # Initialize RAG pipeline
        self.rag_pipeline = RAGPipeline(
            retriever=retriever,
            temperature=0.4,
            max_tokens=500,
            verbose=self.verbose
        )
        self.rag_pipeline.initialize()

    def predict_ckd(self, form_data):
        """
        Make CKD prediction and generate personalized advice.

        Args:
            form_data: Dictionary with health parameters

        Returns:
            tuple: (prediction_result, advice)
        """
        # Preprocess and predict
        features = preprocess_input(form_data)
        prediction = self.ml_pipeline.model.predict(features)[0]

        # Generate context-based query for RAG
        if prediction == 1:
            query = (
                "The user is at risk of kidney disease. "
                "What advice can you provide to help them manage this risk?"
            )
            result = 'Kidney disease likely'
        else:
            query = (
                "The user is not at immediate risk of kidney disease. "
                "What advice can you provide to help them maintain good kidney health?"
            )
            result = 'Kidney disease unlikely'

        # Get personalized advice from RAG
        rag_response = self.rag_pipeline.query(query)
        advice = rag_response.get("answer", "I'm sorry, I couldn't find an answer to your question.")

        advice = self._clean_response(advice)

        return result, advice

    def _clean_response(self, text):
        """Remove source tags, KidneyCareAI prefix, and leading artifacts from LLM response."""
        text = text.replace("[SOURCES_USED]", "").replace("[NO_SOURCES]", "")
        text = text.replace("`[SOURCES_USED]`", "").replace("`[NO_SOURCES]`", "")
        text = text.replace("KidneyCareAI:", "")
        text = text.strip()
        # Remove leading stray punctuation left behind after tag stripping
        text = text.lstrip("?!.,;:- \n")
        # Re-strip whitespace after removing leading punctuation
        return text.strip()

    def chat(self, message):
        """
        Handle chatbot interaction.

        Args:
            message: User's message string

        Returns:
            dict: {'answer': str, 'sources': list}
        """
        # Query RAG pipeline
        rag_response = self.rag_pipeline.query(message)
        raw_answer = rag_response.get("answer", "I'm sorry, I couldn't find an answer to your question.")
        context_docs = rag_response.get("context", [])

        # Detect whether LLM used sources before cleaning tags
        sources_used = '[SOURCES_USED]' in raw_answer

        # Clean the answer (strips tags and artifacts)
        cleaned_answer = self._clean_response(raw_answer)

        # Extract source metadata from retrieved documents
        sources = []
        if sources_used:
            seen = set()
            for doc in context_docs:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                source_name = metadata.get('source', 'Unknown')
                if '\\' in source_name or '/' in source_name:
                    source_name = source_name.split('\\')[-1].split('/')[-1]
                page = metadata.get('page', 'N/A')
                # Convert 0-indexed page to 1-indexed for display
                if isinstance(page, int):
                    page = page + 1
                key = (source_name, page)
                if key not in seen:
                    seen.add(key)
                    preview = doc.page_content[:100] + '...' if hasattr(doc, 'page_content') else ''
                    sources.append({'source': source_name, 'page': page, 'preview': preview})

        return {'answer': cleaned_answer, 'sources': sources}


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ckd-chatbot-secret-key-2024')

# Global chatbot instance (singleton pattern)
chatbot_core = None


def get_chatbot():
    """Get or create the CKDChatbotCore instance."""
    global chatbot_core
    if chatbot_core is None:
        chatbot_core = CKDChatbotCore(verbose=True)
    return chatbot_core


@app.route('/')
def home():
    """Render the main chat interface."""
    return render_template('index.html')


@app.route('/health-form')
def health_form():
    """Render the health details form."""
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle ML prediction request."""
    try:
        # Get form data
        data = request.form.to_dict()

        # Get chatbot instance
        chatbot = get_chatbot()

        # Make prediction and get advice
        prediction_result, advice = chatbot.predict_ckd(data)

        # Return result
        return render_template('result.html', prediction=prediction_result, advice=advice)

    except Exception as e:
        print(f"Error in /predict: {e}")
        return render_template('result.html',
                             prediction="Error",
                             advice=f"An error occurred: {str(e)}")


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chatbot conversation."""
    try:
        # Access JSON payload
        data = request.json
        if not data:
            return jsonify({"response": "No JSON payload received."}), 400

        # Extract user message
        user_message = data.get('message')
        if not user_message:
            return jsonify({"response": "No message provided in the request."}), 400

        # Simulate typing delay (2 seconds)
        time.sleep(2)

        # Get chatbot instance
        chatbot = get_chatbot()

        # Get bot response with sources
        result = chatbot.chat(user_message)

        # Return response as JSON
        return jsonify({
            "response": result['answer'],
            "sources": result['sources'],
            "sources_count": len(result['sources'])
        })

    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # Initialize the chatbot on startup
    get_chatbot()

    # Run Flask development server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
