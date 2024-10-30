'''from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import numpy as np
from model import preprocess_input
import requests

# Load the trained model for kidney disease prediction
model = joblib.load('./data/kidney_disease_rf_model.pkl')

# Hugging Face API information
HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HUGGING_FACE_API_KEY = "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" 

headers = {"Authorization": f"Bearer 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"}

# Initialize the Flask app
app = Flask(__name__)

# Home route to display the chatbot and link to the form
@app.route('/')
def home():
    return render_template('index.html')

# Link to health details form
@app.route('/health-form')
def health_form():
    return render_template('form.html')

# Prediction route to handle form submission and predict kidney disease
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()  # Get form data as a dictionary
    features = preprocess_input(data)  # Preprocess user input
    prediction = model.predict(features)[0]  # Predict using the preprocessed features
    
    # Prepare the prediction result and personalized advice
    if prediction == 1:
        result = 'Kidney disease likely'
        advice = """
        Based on your results, here’s some advice:
        1. Consult a healthcare professional for further tests.
        2. Follow a kidney-friendly diet, monitor blood pressure, and stay hydrated.
        3. Consider medications for managing diabetes or hypertension.
        """
    else:
        result = 'Kidney disease unlikely'
        advice = """
        Your kidneys appear to be functioning well. However, to maintain good kidney health:
        1. Stay hydrated, limit salt intake, and exercise regularly.
        2. Monitor your blood pressure and blood sugar levels.
        """
    
    # Render the result along with personalized advice
    return render_template('result.html', prediction=result, advice=advice)

# Chatbot route to handle conversation with the user
@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    # Call Hugging Face API for Mistral-7B model
    response = requests.post(
        HUGGING_FACE_API_URL,
        headers=headers,
        json={"inputs": user_message}
    )
    
    # Extract the response from Hugging Face API
    response_data = response.json()
    # Check if the response is a list and get the first item
    if isinstance(response_data, list) and len(response_data) > 0:
        bot_message = response_data[0]['generated_text']
    else:
        bot_message = "Sorry, I couldn't generate a response."
    
    return jsonify({'response': bot_message})

if __name__ == '__main__':
    app.run(debug=True)







from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import numpy as np
from model import preprocess_input
from langchain_huggingface import HuggingFaceEndpoint
import os

# Load the trained model for kidney disease prediction
model = joblib.load('./data/kidney_disease_rf_model.pkl')

# HuggingFace API configuration
hf_token = "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"  # Replace with your token
REPO_ID = 'mistralai/Mistral-7B-Instruct-v0.3'

# Initialize LangChain HuggingFace endpoint
llm = HuggingFaceEndpoint(
    repo_id=REPO_ID,
    max_length=512,
    temperature=0.7,
    token=hf_token
)

# Initialize the Flask app
app = Flask(__name__)

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
    
    if prediction == 1:
        result = 'Kidney disease likely'
        advice = """
        Based on your results, here's some advice:
        1. Consult a healthcare professional for further tests.
        2. Follow a kidney-friendly diet, monitor blood pressure, and stay hydrated.
        3. Consider medications for managing diabetes or hypertension.
        """
    else:
        result = 'Kidney disease unlikely'
        advice = """
        Your kidneys appear to be functioning well. However, to maintain good kidney health:
        1. Stay hydrated, limit salt intake, and exercise regularly.
        2. Monitor your blood pressure and blood sugar levels.
        """
    
    return render_template('result.html', prediction=result, advice=advice)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json['message']
        
        # Create a more conversational prompt that maintains medical context
        prompt = f"""<s>[INST] <<SYS>>
You are a friendly and professional medical assistant with expertise in kidney health and related medical topics.
- Respond naturally to greetings and casual conversation
- For medical questions, provide accurate and helpful information
- Always maintain a compassionate and professional tone
- For serious medical concerns, recommend consulting a healthcare professional
- Keep responses concise but informative
<</SYS>>

{user_message} [/INST]"""
        
        # Get response from Llama
        response = llm.invoke(prompt)
        
        # Clean up the response
        bot_message = response.strip()
        
        # Ensure we have a response
        if not bot_message:
            bot_message = "Hello! I'm your medical assistant. How can I help you today?"
            
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        bot_message = "I apologize, but I'm experiencing technical difficulties. Please try again later."
    
    return jsonify({'response': bot_message})

if __name__ == '__main__':
    app.run(debug=True)
'''







from flask import Flask, request, jsonify, render_template, redirect, url_for
import joblib
import numpy as np
from model import preprocess_input
import requests
import json

# Load the trained model for kidney disease prediction
model = joblib.load('./data/kidney_disease_rf_model.pkl')

# Hugging Face API information
HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HUGGING_FACE_API_KEY = "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"

headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

# Initialize the Flask app
app = Flask(__name__)

# Predefined context and medical information
MEDICAL_CONTEXT = """
You are a helpful medical assistant chatbot specializing in kidney health. Your role is to:
1. Provide general information about kidney health and disease
2. Answer questions about kidney-related symptoms and prevention
3. Offer lifestyle advice for maintaining kidney health
4. Recommend when to seek professional medical help
5. Explain common kidney tests and procedures

Remember to:
- Always emphasize that you're an AI assistant and not a replacement for professional medical advice
- Be empathetic and supportive while maintaining professionalism
- Provide clear, accurate information based on medical knowledge
- Encourage users to consult healthcare professionals for specific medical advice
"""

# Common kidney-related topics and responses
KIDNEY_INFO = {
    "symptoms": [
        "frequent urination",
        "blood in urine",
        "foamy urine",
        "swelling in feet and ankles",
        "fatigue",
        "high blood pressure"
    ],
    "risk_factors": [
        "diabetes",
        "high blood pressure",
        "obesity",
        "family history",
        "age over 60",
        "smoking"
    ],
    "prevention_tips": [
        "stay hydrated",
        "maintain healthy blood pressure",
        "control blood sugar levels",
        "eat a balanced diet",
        "exercise regularly",
        "limit alcohol intake"
    ]
}

def generate_prompt(user_message):
    """Generate a structured prompt for the language model"""
    return f"""Context: {MEDICAL_CONTEXT}

Previous message: {user_message}

Please provide a helpful, accurate, and empathetic response while:
1. Addressing the user's specific concerns
2. Providing relevant medical information if appropriate
3. Encouraging professional medical consultation when necessary
4. Using clear and simple language

Response:"""

def analyze_message_intent(message):
    """Analyze the user's message to determine the primary intent"""
    message = message.lower()
    
    if any(symptom in message for symptom in KIDNEY_INFO["symptoms"]):
        return "symptom_inquiry"
    elif any(factor in message for factor in KIDNEY_INFO["risk_factors"]):
        return "risk_assessment"
    elif any(word in message for word in ["prevent", "healthy", "advice", "tips"]):
        return "prevention_advice"
    elif any(word in message for word in ["test", "diagnosis", "doctor", "hospital"]):
        return "medical_guidance"
    else:
        return "general_inquiry"

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
    
    if prediction == 1:
        result = 'Kidney disease likely'
        advice = """
        Based on your results, here's some advice:
        1. Please schedule an appointment with a healthcare provider for proper evaluation.
        2. Monitor your blood pressure and blood sugar levels regularly.
        3. Follow a kidney-friendly diet:
           - Reduce salt intake
           - Control protein consumption
           - Stay hydrated with appropriate fluid intake
        4. Keep track of any symptoms you experience.
        5. Prepare questions for your healthcare provider.
        """
    else:
        result = 'Kidney disease unlikely'
        advice = """
        Your kidneys appear to be functioning well. To maintain good kidney health:
        1. Continue regular exercise and maintain a healthy weight
        2. Follow a balanced diet with moderate salt intake
        3. Stay well-hydrated with 6-8 glasses of water daily
        4. Monitor blood pressure and blood sugar levels
        5. Schedule regular check-ups with your healthcare provider
        """
    
    return render_template('result.html', prediction=result, advice=advice)

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    
    # Analyze message intent
    intent = analyze_message_intent(user_message)
    
    # Generate appropriate prompt based on intent
    prompt = generate_prompt(user_message)
    
    try:
        # Call Hugging Face API
        response = requests.post(
            HUGGING_FACE_API_URL,
            headers=headers,
            json={"inputs": prompt}
        )
        
        response_data = response.json()
        
        if isinstance(response_data, list) and len(response_data) > 0:
            bot_message = response_data[0]['generated_text']
            
            # Clean up the response to remove the prompt and context
            bot_message = bot_message.split("Response:")[-1].strip()
        else:
            bot_message = """I apologize, but I'm having trouble generating a response. 
            Please rephrase your question or consult a healthcare professional for medical advice."""
            
    except Exception as e:
        bot_message = """I apologize for the technical difficulty. Please try again or reach out to a healthcare provider for immediate assistance."""
        
    return jsonify({'response': bot_message})

if __name__ == '__main__':
    app.run(debug=True)


