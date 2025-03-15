import os
import pickle
from flask import Flask, render_template, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Get the base directory of the script (app.py)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved model and vectorizer using the absolute paths
model_path = os.path.join(base_dir, 'spam_classifier', 'model.pkl')
vectorizer_path = os.path.join(base_dir, 'spam_classifier', 'vectorizer.pkl')

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

# Load the pre-trained T5 model and tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")
t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Function to paraphrase the spam message using the T5 model
def paraphrase_spam(message):
    input_text = f"paraphrase: {message} </s>"
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate paraphrased output with beam search
    outputs = t5_model.generate(
        inputs, 
        max_length=150, 
        num_return_sequences=1, 
        num_beams=5, 
        no_repeat_ngram_size=2, 
        early_stopping=True
    )
    
    paraphrased_message = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return paraphrased_message

# Function to format email properly
def format_email(subject, body):
    formatted_email = f"""
    <strong>Subject:</strong> {subject} <br><br>
    <strong>Body:</strong> <br>
    <blockquote>{body}</blockquote>
    """
    return formatted_email

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    data = [message]
    vectorized_data = vectorizer.transform(data)
    prediction = model.predict(vectorized_data)[0]

    # Dummy example for email structure extraction (Subject, Body)
    subject = "Sample Email Subject"  # In real usage, extract subject from message
    body = message  # In real usage, extract the body part of the email

    if prediction == 1:  # Spam
        corrected_message = paraphrase_spam(body)  # Use paraphrasing function here
        result = "Spam"
        formatted_corrected_email = format_email(subject, corrected_message)
        advice = formatted_corrected_email
    else:
        result = "Ham"
        advice = "This message seems fine."

    return jsonify({
        'prediction': result,
        'corrected_message': advice
    })

if __name__ == '__main__':
    app.run(debug=True)
