import tensorflow as tf
import torch
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import classification_report
from IMDBmovieset import X_test_encoded, y_test
import torch.nn.functional as F

path = "C:\\Users\\lenovo\\Desktop\\on-going projects\\BERTmodel2"

bert_tokenizer = BertTokenizer.from_pretrained(path + '\\Tokenizer')
bert_model = TFBertForSequenceClassification.from_pretrained(path + '\\Model')

# Predict the sentiment of the test dataset
pred = bert_model.predict(
    [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']])
# pred is of type TFSequenceClassifierOutput
logits_np = pred.logits
# convert the numpy array to a pytorch tensor
logits = torch.from_numpy(logits_np)

# Apply softmax to obtain probabilities
probabilities = F.softmax(logits, dim=-1)

# get the confidence score
confidence_score = probabilities.max().item()

# Use argmax along the appropriate axis to get the predicted labels
pred_labels = tf.argmax(logits, axis=1)

# Convert the predicted labels to a NumPy array
pred_labels = pred_labels.numpy()

label = {
    1: 'positive',
    0: 'Negative'
}

# Map the predicted labels to their corresponding strings using the label dictionary
pred_labels = [label[i] for i in pred_labels]
Actual = [label[i] for i in y_test]

print('Predicted Label :', pred_labels[:10])
print('Actual Label    :', Actual[:10])

print('Predicted Label :', pred_labels[:10])
print('Actual Label    :', Actual[:10])

print("Classification Report: \n", classification_report(Actual, pred_labels))
print("Confidence score: ", confidence_score)
