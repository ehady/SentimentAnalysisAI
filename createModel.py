import os
from nltk.corpus import stopwords
import shutil
import tarfile
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objects as go
# from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Get the current working directory
current_folder = os.getcwd()

# dataset = tf.keras.utils.get_file(
    # fname="aclImdb.tar.gz",
    # origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    # cache_dir=current_folder,
    # extract=True)

# dataset_path = os.path.dirname(dataset)
# Check the dataset
# os.listdir(dataset_path)
train_negative_folder = "C:\\Users\\lenovo\\PycharmProjects\\BERTdenemesi\\datasets\\aclImdb\\train\\neg"
train_positive_folder = "C:\\Users\\lenovo\\PycharmProjects\\BERTdenemesi\\datasets\\aclImdb\\train\\pos"
test_folder = "C:\\Users\\lenovo\\Downloads\\aclImdb_v1\\aclImdb\\test"
test_pos_folder = "C:\\Users\\lenovo\\PycharmProjects\\BERTdenemesi\\datasets\\aclImdb\\test\\pos"
test_neg_folder = "C:\\Users\\lenovo\\PycharmProjects\\BERTdenemesi\\datasets\\aclImdb\\test\\neg"
# Dataset directory
dataset_dir = "C:\\Users\\lenovo\\PycharmProjects\\BERTdenemesi\\datasets"
# Train dataset folder
train_dir = "C:\\Users\\lenovo\\PycharmProjects\\BERTdenemesi\\datasets\\aclImdb\\train"


# Read the files of the "Train" directory files

# for file in os.listdir(train_dir):
# file_path = os.path.join(train_dir, file)
# Check if it's a file (not a directory)
# if os.path.isfile(file_path):
# with open(file_path, 'r', encoding='utf-8') as f:
# first_value = f.readline().strip()
# print(f"{file}: {first_value}")
# else:
# print(f"{file}: {file_path}")

# -----THIS PART WORKS ---

# Load the movies reviews and convert them into pandas' dataframe with respected sentiments
# 1 means pos 0 means neg

def load_dataset(directory):
    data = {"sentence": [], "sentiment": []}
    for file_name in os.listdir(directory):
        print(file_name)
        if file_name == 'pos':
            positive_dir = os.path.join(directory, file_name)
            for text_file in os.listdir(positive_dir):
                text = os.path.join(positive_dir, text_file)
                with open(text, "r", encoding="utf-8") as f:
                    data["sentence"].append(f.read())
                    data["sentiment"].append(1)
        elif file_name == 'neg':
            negative_dir = os.path.join(directory, file_name)
            for text_file in os.listdir(negative_dir):
                text = os.path.join(negative_dir, text_file)
                with open(text, "r", encoding="utf-8") as f:
                    data["sentence"].append(f.read())
                    data["sentiment"].append(0)

    return pd.DataFrame.from_dict(data)


# Load the dataset from the train_dir
train_df = load_dataset(train_dir)

print(train_df.head())

test_dir = "C:\\Users\\lenovo\\PycharmProjects\\BERTdenemesi\\datasets\\aclImdb\\test"

# Load the dataset from the train_dir
test_df = load_dataset(test_dir)
print(test_df.head())

sentiment_counts = train_df['sentiment'].value_counts()

fig = px.bar(x={0: 'Negative', 1: 'Positive'},
             y=sentiment_counts.values,
             color=sentiment_counts.index,
             color_discrete_sequence=px.colors.qualitative.Dark24,
             title='<b>Sentiments Counts')

fig.update_layout(title='Sentiments Counts',
                  xaxis_title='Sentiment',
                  yaxis_title='Counts',
                  template='plotly_dark')

# Show the bar chart
fig.show()
pyo.plot(fig, filename='Sentiments Counts.html', auto_open=True)

def text_cleaning(text):
    soup = BeautifulSoup(text, "html.parser")
    text = re.sub(r'\[[^]]*\]', '', soup.get_text())
    pattern = r"[^a-zA-Z0-9\s,']"
    text = re.sub(pattern, '', text)
    return text

# Train dataset
train_df['Cleaned_sentence'] = train_df['sentence'].apply(text_cleaning).tolist()
# Test dataset
test_df['Cleaned_sentence'] = test_df['sentence'].apply(text_cleaning)

# Training data
# Reviews = "[CLS] " +train_df['Cleaned_sentence'] + "[SEP]"
Reviews = train_df['Cleaned_sentence']
Target = train_df['sentiment']

# Test data
# test_reviews =  "[CLS] " +test_df['Cleaned_sentence'] + "[SEP]"
test_reviews = test_df['Cleaned_sentence']
test_targets = test_df['sentiment']

x_val, x_test, y_val, y_test = train_test_split(test_reviews,
                                                    test_targets,
                                                    test_size=0.5,
                                                    stratify = test_targets)

#Tokenize and encode the data using the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

max_len = 128
# Tokenize and encode the sentences
X_train_encoded = tokenizer.batch_encode_plus(Reviews.tolist(),
                                              padding=True,
                                              truncation=True,
                                              max_length=max_len,
                                              return_tensors='tf')

X_val_encoded = tokenizer.batch_encode_plus(x_val.tolist(),
                                            padding=True,
                                            truncation=True,
                                            max_length=max_len,
                                            return_tensors='tf')

X_test_encoded = tokenizer.batch_encode_plus(x_test.tolist(),
                                             padding=True,
                                             truncation=True,
                                             max_length=max_len,
                                             return_tensors='tf')

k = 0
print('Training Comments -->>',Reviews[k])
print('\nInput Ids -->>\n',X_train_encoded['input_ids'][k])
print('\nDecoded Ids -->>\n',tokenizer.decode(X_train_encoded['input_ids'][k]))
print('\nAttention Mask -->>\n',X_train_encoded['attention_mask'][k])
print('\nLabels -->>',Target[k])

# Intialize the model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Compile the model with an appropriate optimizer, loss function, and metrics
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Step 5: Train the model
history = model.fit(
    [X_train_encoded['input_ids'], X_train_encoded['token_type_ids'], X_train_encoded['attention_mask']],
    Target,
    validation_data=(
      [X_val_encoded['input_ids'], X_val_encoded['token_type_ids'], X_val_encoded['attention_mask']],y_val),
    batch_size=32,
    epochs=3
)
#Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(
    [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']],
    y_test
)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

path = 'C:\\Users\\lenovo\\Desktop\\on-going projects\\BERTmodel2'
# Save tokenizer
tokenizer.save_pretrained(path + '/Tokenizer')

# Save model
model.save_pretrained(path + '/Model')