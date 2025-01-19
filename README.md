# Chronic Kidney Disease Prediction and Management Chatbot

## Overview
The **Chronic Kidney Disease Prediction and Management Chatbot** is an interactive application designed to provide users with information and assistance regarding chronic kidney disease (CKD). 
This chatbot leverages natural language processing to engage users in conversation and deliver relevant health information. 
Additionally, it features a Random Forest prediction model that classifies CKD and non-CKD cases. This project seamlessly combine the CKD prediction model with the chatbot to provide personalized, actionable insights during patient interactions

## Features
- **User-Friendly Interface**: Intuitive design for easy interaction.
- **Information Retrieval (RAG)**: Provides accurate information about chronic kidney disease.
- **Machine Learning Integration**: Utilizes a Random Forest model for classifying CKD and non-CKD cases.

## Technologies Used
- **Python**: Core programming language for backend logic.
- **HTML/CSS/JavaScript**: Frontend development for user interface.
- **Flask**: Web framework for building the application.
- **Natural Language Processing (Openai LLM)**: For understanding user queries.
- **Random Forest**: Machine learning algorithm for classification of CKD cases.

## Installation

To set up the project locally, follow these steps:

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

3. **Run the application**:
   Start the Flask server:
   ```bash
   python app.py
   ```

4. **Access the chatbot**:
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


## Contact
For questions or suggestions, please reach out to [engrstephdez@gmail.com](mailto:engrstephdez@gmail.com).
