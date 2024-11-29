from flask import Flask, render_template, jsonify, request, redirect, url_for
import joblib
import numpy as np
from model import preprocess_input
import requests
import json
from store import download_hugging_face_embeddings
from langchain_pinecone import Pinecone
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
#from helper import download_hugging_face_embeddings
from helper import load_cached_embeddings
import os
import time

# Load the trained model for kidney disease prediction
model = joblib.load('./data/kidney_disease_rf_model.pkl')

# Initialize the Flask app
app = Flask(__name__)



chat_prompt = (
    "You are KidneyCareAI, a compassionate and knowledgeable medical assistant chatbot specializing in chronic kidney disease (CKD) prediction, management, and education. "
    "You leverage retrieval-augmented generation (RAG) to provide users with accurate, personalized, and empathetic responses. "
    "Your primary objectives are:\n"
    "1. Assist users in understanding their CKD risk and prediction results with clear, user-friendly explanations.\n"
    "2. Generate personalized advice for CKD management, including lifestyle, diet, and medical recommendations, tailored to the user's specific circumstances.\n"
    "3. Educate users about CKD symptoms, causes, prevention, and treatment options in an accessible manner.\n"
    "4. Respond empathetically and professionally, encouraging users to seek medical advice when needed.\n\n"
    "Always retrieve and incorporate relevant information to ensure accurate and contextually appropriate responses. If you don't know the answer, acknowledge it politely and encourage the user to consult a healthcare professional.\n\n"
    "Respond to the user's queries conversationally, empathetically, and concisely. Tailor your responses to their specific needs while clearly stating you are not a replacement for professional medical advice."
    "\n\n{context}"
)

# Load environment variables
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load Hugging Face embeddings (using caching)
embeddings = load_cached_embeddings()
#embeddings = download_hugging_face_embeddings()
docsearch = Pinecone.from_existing_index(index_name='ckd-chatbot', embedding=embeddings)



index_name = "ckd-chatbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


llm = OpenAI(temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", chat_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health-form')
def health_form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    features = preprocess_input(data)
    prediction = model.predict(features)[0]
    
    # Create context for RAG
    if prediction == 1:
        query = (
            "The user is at risk of kidney disease"
            "What advice can you provide to help them manage this risk?"
        )
        result = 'Kidney disease likely'
    else:
        query = (
            "The user is not at immediate risk of kidney disease"
            "What advice can you provide to help them maintain good kidney health?"
        )
        result = 'Kidney disease unlikely'
    
    # Use RAG to generate personalized advice
    rag_response = rag_chain.invoke({"input": query})
    advice = rag_response.get("answer", "I'm sorry, I couldn't find an answer to your question.")
    advice = advice.replace("KidneyCareAI:", "").strip()  # Remove any unnecessary prefix

    # Return prediction and personalized advice
    return render_template('result.html', prediction=result, advice=advice)




@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Access JSON payload
        data = request.json
        if not data:
            return jsonify({"response": "No JSON payload received."}), 400

        # Extract user message
        user_message = data.get('message')
        if not user_message:
            return jsonify({"response": "No message provided in the request."}), 400

        # Simulate typing delay
        time.sleep(2)  # Add a 2-second delay to mimic typing

        # Invoke RAG chain to get the chatbot's response
        rag_response = rag_chain.invoke({"input": user_message})

        # Extract the final answer without redundant prefixes
        bot_response = rag_response.get("answer", "I'm sorry, I couldn't find an answer to your question.")
        bot_response = bot_response.replace("KidneyCareAI:", "").strip()  # Remove any unnecessary prefix

        # Return the response as JSON
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"response": f"An error occurred: {str(e)}"}), 500



if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Default to port 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
