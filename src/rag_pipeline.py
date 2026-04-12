"""RAG Pipeline and LLM Integration"""

from openai import OpenAI as OpenAIClient


# ---------------------------------------------------------------------------
# System prompt — defines KidneyCareAI persona and source-tagging protocol
# ---------------------------------------------------------------------------

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
- NEVER make up medical facts not in the retrieved information"""


# User prompt template — injects retrieved context and the user's question
_USER_PROMPT_TEMPLATE = """Here is relevant medical information retrieved from clinical literature:

{context}

---

User's question: {question}

---

Please respond as KidneyCareAI. Use the reference information above to inform your answer,
but explain it in your own words in a warm, patient-friendly tone.
Remember to start with [SOURCES_USED] or [NO_SOURCES] as instructed."""


class RAGPipeline:
    """
    RAG pipeline for KidneyCareAI chatbot.

    Uses HybridRetriever for document retrieval and calls the OpenAI
    chat completions API directly for response generation.
    """

    def __init__(
        self,
        retriever,
        model: str = "gpt-4o-mini",
        temperature: float = 0.4,
        max_tokens: int = 500,
        verbose: bool = True
    ):
        """
        Initialize RAG pipeline.

        Args:
            retriever   : HybridRetriever instance
            model       : OpenAI chat model name
            temperature : Sampling temperature (default: 0.4)
            max_tokens  : Maximum tokens in response (default: 500)
            verbose     : Print progress information
        """
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.client: OpenAIClient | None = None

    def initialize(self):
        """Create the OpenAI client."""
        if self.verbose:
            print("Initializing RAG pipeline...")

        self.client = OpenAIClient()

        if self.verbose:
            print(f"RAG pipeline initialized — model: {self.model}")

    def query(self, question: str) -> dict:
        """
        Run a full RAG query: retrieve → build context → generate response.

        Args:
            question: User's question string

        Returns:
            dict with keys:
              'answer'  : str — the LLM-generated response (may include source tags)
              'context' : List[Document] — retrieved chunks (used by app.py for citations)
        """
        if self.client is None:
            raise ValueError("RAG pipeline not initialized. Call initialize() first.")

        # 1. Hybrid retrieval
        docs = self.retriever.retrieve(question)

        # 2. Build context string from retrieved chunks
        if docs:
            context = "\n\n".join(doc.page_content for doc in docs)
        else:
            context = "No relevant context found in the knowledge base."

        # 3. Call OpenAI chat completions
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )

        messages = [
            {"role": "system", "content": KIDNEYCAREAI_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = response.choices[0].message.content

        return {"answer": answer, "context": docs}

    def get_answer(self, question: str) -> str:
        """Convenience wrapper — returns just the answer string."""
        return self.query(question).get(
            "answer", "I'm sorry, I couldn't find an answer to your question."
        )
