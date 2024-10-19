from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the saved model and tokenizer
model = load_model('toxic_comment_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Tokenizer and preprocessing should be consistent with your training
MAX_SEQUENCE_LENGTH = 200

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the comment from the form
    comment = request.form['comment']
    
    # Preprocess the comment
    sequences = tokenizer.texts_to_sequences([comment]) 
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # Predict toxicity scores
    prediction = model.predict(data)[0]

    # Define thresholds for blocking
    thresholds = {
        'toxic': 0.4,
        'severe_toxic': 0.3,
        'obscene': 0.2301,
        'threat': 0.2,
        'insult': 0.4,
        'identity_hate': 0.3
    }

    result = "OK"
    for i, label in enumerate(thresholds):
        if prediction[i] > thresholds[label]:
            result = "Blocked by admin"
            break

    return render_template('index.html', comment=comment, result=result)

if __name__ == '__main__':
    app.run(debug=True)
