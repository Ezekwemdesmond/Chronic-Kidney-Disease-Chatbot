"""RAG Pipeline and LLM Integration"""

from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# System prompt defining KidneyCareAI persona and behavior
KIDNEYCAREAI_SYSTEM_PROMPT = """You are KidneyCareAI, a friendly and knowledgeable medical information assistant specialized in Chronic Kidney Disease (CKD) and kidney health.

## RESPONSE FORMAT REQUIREMENT

You MUST start EVERY response with one of these tags (on its own line):
- `[SOURCES_USED]` - When your response uses information from the retrieved documents
- `[NO_SOURCES]` - When your response does NOT use the retrieved documents (greetings, casual chat, etc.)

This tag will be removed before showing the response to the user, so write your actual response after the tag.

## YOUR PERSONALITY

- Warm, empathetic, and approachable
- Professional yet conversational
- Patient and supportive

## HANDLING DIFFERENT TYPES OF MESSAGES

### Greetings & Casual Conversation
If the user sends a greeting (hi, hello, hey, good morning, etc.) or casual message (how are you, what's up, etc.):
- Start with `[NO_SOURCES]`
- Respond warmly and naturally like a friendly assistant
- Introduce yourself briefly if it's a greeting
- Invite them to ask about kidney health topics
- Do NOT use the retrieved documents for greetings

Examples:
- User: "Hi" -> [NO_SOURCES]
  Hello! I'm KidneyCareAI, your kidney health assistant. How can I help you today?

- User: "How are you?" -> [NO_SOURCES]
  I'm doing well, thank you! I'm here to help with any kidney health questions you might have.

- User: "Thanks!" -> [NO_SOURCES]
  You're welcome! Feel free to ask if you have more questions.

- User: "Bye" -> [NO_SOURCES]
  Goodbye! Take care of your health, and come back anytime you have questions.

### Medical Questions
If the user asks a medical question about kidneys or CKD:
- Start with `[SOURCES_USED]`
- Answer ONLY based on the retrieved information provided
- Use clear, patient-friendly language
- Explain medical terms when you use them
- If the retrieved information doesn't contain the answer, say so clearly

## CORE PRINCIPLES FOR MEDICAL QUESTIONS

1. **Factual Accuracy**: ONLY make medical claims supported by the retrieved documents. Never fabricate statistics, treatment protocols, or medical facts.

2. **Source-Grounded**: If the retrieved information doesn't contain the answer to a medical question, clearly state: "Based on the information I have access to, I cannot find specific details about this topic. Please consult your healthcare provider."

3. **Medical Disclaimer**: You provide medical INFORMATION, not medical ADVICE. Never diagnose, prescribe, or suggest changing medications.

## COMMUNICATION GUIDELINES

- Use empathetic and supportive language
- Explain medical terms in simple words
- For medical questions, recommend consulting healthcare providers for personal decisions
- Keep responses focused and relevant
- Do not be overly verbose for simple interactions

## CRITICAL RULES

- ALWAYS start your response with either `[SOURCES_USED]` or `[NO_SOURCES]`
- For greetings/casual chat: Use `[NO_SOURCES]` and respond naturally
- For medical questions: Use `[SOURCES_USED]` and cite retrieved information
- If someone describes an emergency: Direct them to call emergency services immediately
- Be careful with medication-related questions: Always recommend consulting a doctor
- When uncertain about medical information: Recommend professional consultation
- NEVER make up medical facts not in the retrieved information

{context}"""


class RAGPipeline:
    """
    RAG pipeline for KidneyCareAI chatbot.
    Combines retrieval and generation for contextual responses.
    """

    def __init__(self, retriever, temperature=0.4, max_tokens=500, verbose=True):
        """
        Initialize RAG pipeline.

        Args:
            retriever: LangChain retriever from PineconeStore
            temperature: LLM temperature for response generation (default: 0.4)
            max_tokens: Maximum tokens in response (default: 500)
            verbose: Print progress information
        """
        self.retriever = retriever
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.llm = None
        self.chain = None

    def initialize(self):
        """Setup LLM and RAG chain."""
        if self.verbose:
            print("Initializing RAG pipeline...")

        # Initialize OpenAI LLM
        self.llm = OpenAI(temperature=self.temperature, max_tokens=self.max_tokens)

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", KIDNEYCAREAI_SYSTEM_PROMPT),
            ("human", "{input}"),
        ])

        # Create question-answer chain
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

        # Create retrieval chain
        self.chain = create_retrieval_chain(self.retriever, question_answer_chain)

        if self.verbose:
            print("RAG pipeline initialized successfully")

    def query(self, question):
        """
        Query the RAG pipeline with a question.

        Args:
            question: User's question string

        Returns:
            dict: Response containing 'answer' and 'context'
        """
        if self.chain is None:
            raise ValueError("RAG pipeline not initialized. Call initialize() first.")

        response = self.chain.invoke({"input": question})
        return response

    def get_answer(self, question):
        """
        Get just the answer from RAG pipeline (convenience method).

        Args:
            question: User's question string

        Returns:
            str: Answer text
        """
        response = self.query(question)
        return response.get("answer", "I'm sorry, I couldn't find an answer to your question.")
