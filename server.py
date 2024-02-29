from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

app = Flask(__name__)

path = "C:\\Users\\lenovo\\Desktop\\on-going projects\\BERTmodel2"

bert_tokenizer = BertTokenizer.from_pretrained(path + '\\Tokenizer')
bert_model = TFBertForSequenceClassification.from_pretrained(path + '\\Model')

label = {
    1: 'positive',
    0: 'Negative'
}


# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# ... (existing code)

def run_sentiment_analysis(text):
    print("Input Text:", text)
    Input_ids, Token_type_ids, Attention_mask = bert_tokenizer.batch_encode_plus(
        [text],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='tf'
    ).values()

    prediction = bert_model.predict([Input_ids, Token_type_ids, Attention_mask])
    pred_labels = tf.argmax(prediction.logits, axis=1).numpy().tolist()
    result = label[pred_labels[0]]

    print("Predicted Result:", result)

    return result


# Route for emotion detection
@app.route('/emotionDetector', methods=['POST'])
def emotion_detector():
    if request.method == 'POST':
        data = request.get_json()
        text_to_analyze = data.get('text')

        result = run_sentiment_analysis(text_to_analyze)

        return jsonify({"result": result})


# def get_sentiment(input, Tokenizer=bert_tokenizer, Model=bert_model):
# Convert input into a list if it's not already a list
# if not isinstance(input,list):
#   input = [input]

# Input_ids, Token_type_ids, Attention_mask = Tokenizer.batch_encode_plus(input,padding=True,
# truncation=True,
# max_length=128,
# return_tensors='tf').values()
# prediction = Model.predict([Input_ids, Token_type_ids, Attention_mask])
# Use argmax along the appropriate axis to get the predicted labels
# pred_labels = tf.argmax(prediction.logits, axis=1)
# Convert the TensorFlow tensor to a NumPy array and then to a list to get the predicted sentiment labels
# pred_labels = [label[i] for i in pred_labels.numpy().tolist()]
# return pred_labels


if __name__ == '__main__':
    app.run(debug=True)
