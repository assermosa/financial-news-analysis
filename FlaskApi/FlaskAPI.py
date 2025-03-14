from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertModel  # Import TFBertModel

# Initialize Flask app
app = Flask(__name__)

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the TensorFlow model from the .h5 file
MODEL_PATH = 'C:\Users\DELL\Downloads\MultiTask\models\saved_models\multi_ta.h5'  # Use raw string
custom_objects = {'TFBertModel': TFBertModel}  # Register custom layer
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(MODEL_PATH)

# Define a function for inference
def predict_sentiment(text):
    """
    Predicts the sentiment and rank score for a given input text.
    """
    # Tokenize the input text
    inputs = tokenizer(
        text, 
        max_length=128, 
        truncation=True, 
        padding='max_length', 
        return_tensors='tf',
        return_token_type_ids=False  # Disable token_type_ids
    )
    
    # Perform inference
    outputs = model([inputs['input_ids'], inputs['attention_mask']])
    
    # Extract the sentiment score (regression output)
    sentiment_score = outputs[0].numpy()[0][0]  # Assuming sentiment_score is the first output
    
    # Map the sentiment score to a label
    sentiment_label = "POSITIVE" if sentiment_score > 0.5 else "NEGATIVE"
    
    # Get the rank score (if your model outputs it)
    if len(outputs) > 1:  # Assuming rank_score is the second element in outputs
        rank_score = outputs[1].numpy()[0][0]
    else:
        rank_score = None
    
    return {
        "text": text,
        "sentiment": sentiment_label,
        "sentiment_score": float(sentiment_score),
        "rank_score": float(rank_score) if rank_score is not None else None
    }

# Define the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for sentiment and rank score prediction.
    """
    # Get the JSON data from the request
    data = request.get_json()
    
    # Validate the input
    if not data or 'text' not in data:
        return jsonify({"error": "Input text is required"}), 400
    
    # Get the input text
    text = data['text']
    
    # Perform inference
    result = predict_sentiment(text)
    
    # Return the result as JSON
    return jsonify(result)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)