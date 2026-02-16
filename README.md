# Chronic Kidney Disease Prediction and Management Chatbot

## Overview
The **Chronic Kidney Disease Prediction and Management Chatbot** is an interactive application designed to provide users with information and assistance regarding chronic kidney disease (CKD).
This chatbot leverages natural language processing to engage users in conversation and deliver relevant health information.
Additionally, it features a Random Forest prediction model that classifies CKD and non-CKD cases. This project seamlessly combine the CKD prediction model with the chatbot to provide personalized, actionable insights during patient interactions

## Features
- **User-Friendly Interface**: Intuitive design for easy interaction.
- **Information Retrieval (RAG)**: Provides accurate information about chronic kidney disease.
- **Machine Learning Integration**: Utilizes a Random Forest model for classifying CKD and non-CKD cases.

## 🎥 Demo

![CKD Chatbot Demo](ckd.gif)

## Technologies Used
- **Python**: Core programming language for backend logic.
- **HTML/CSS/JavaScript**: Frontend development for user interface.
- **Flask**: Web framework for building the application.
- **Docker**: Containerization for consistent deployment.
- **Natural Language Processing (Openai LLM)**: For understanding user queries.
- **Random Forest**: Machine learning algorithm for classification of CKD cases.
- **OpenAI API**: For natural language processing capabilities.
- **Pinecone API**: For managing and querying vector embeddings.

## Architecture

This project follows a modular architecture with clean separation of concerns:

### Project Structure
```
src/
├── __init__.py              # Package exports
├── ml_model.py             # ML model training and prediction
├── document_processing.py  # PDF loading and embeddings
├── vectorstore.py          # Pinecone vector store management
└── rag_pipeline.py         # RAG chain and LLM integration

app.py                      # Flask application entry point
Dockerfile                  # Docker container configuration
```

### Key Design Patterns
- **Modular Package Structure**: Business logic organized in `src/` modules
- **Class-Based Components**: Clean encapsulation (MLModelPipeline, PineconeStore, RAGPipeline)
- **Orchestration Pattern**: CKDChatbotCore class coordinates all components
- **Separation of Concerns**: Flask routes separate from business logic

## Installation

### Option 1: Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ezekwemdesmond/Chronic-Kidney-Disease-Chatbot.git
   cd Chronic-Kidney-Disease-Chatbot
   ```

2. **Set up API keys**:
   Create a `.env` file in the project root:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

3. **Build and run with Docker**:
   ```bash
   docker build -t ckd-chatbot .
   docker run -p 5000:5000 --env-file .env ckd-chatbot
   ```

4. **Access the chatbot**:
   Open your web browser and navigate to `http://localhost:5000`.

### Option 2: Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ezekwemdesmond/Chronic-Kidney-Disease-Chatbot.git
   cd Chronic-Kidney-Disease-Chatbot
   ```

2. **Install dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**:
   Create a `.env` file in the project root:
   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Access the chatbot**:
   Open your web browser and navigate to `http://localhost:5000`.

## Usage
Once the application is running, users can interact with the chatbot through the web interface. Simply type in your questions related to chronic kidney disease, and the chatbot will provide appropriate responses. Additionally, users can input relevant health data to receive a classification of CKD or non-CKD based on the Random Forest model and personalised advice.

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## Acknowledgements
- **Dataset**: https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

## Contact
For questions or suggestions, please reach out to [engrstephdez@gmail.com](mailto:engrstephdez@gmail.com).
