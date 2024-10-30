"""def predict():
    data = request.json

    # Preprocess the input for the model
    features = preprocess_input(data)

    # Predict using the pre-trained model
    prediction = model.predict([features])
    result = 'Positive for Kidney Disease' if prediction == 1 else 'Negative for Kidney Disease'

    # Provide personalized advice based on prediction
    advice = get_personalized_advice(prediction)

    return jsonify({'prediction': result, 'advice': advice})
"""
# Ensure numerical columns are converted to float
    for col in numerical_columns:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(float)  # Convert to float