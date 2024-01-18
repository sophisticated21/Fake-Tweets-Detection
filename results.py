import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd 

model = load_model("real_fake_tweets.h5")


with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

def load_clean_tweets(filenames):
    """
    Load clean tweets from CSV files.
    :param filenames: List of filenames to load.
    :return: Concatenated DataFrame of clean tweets.
    """
    tweet_dfs = [pd.read_csv(filename) for filename in filenames]
    all_clean_tweets = pd.concat(tweet_dfs, ignore_index=True)
    return all_clean_tweets

def tokenize_tweets(tokenizer, tweets):
    """
    Convert tweets into padded sequences.
    :param tokenizer: Tokenizer object.
    :param tweets: List or Series of tweets.
    :return: Padded sequences of tokenized tweets.
    """
    sequences = tokenizer.texts_to_sequences(tweets)
    padded_sequences = pad_sequences(sequences, padding='post')
    return padded_sequences

with open('max_length.txt') as f:
    max_length = int(f.read())

filenames = ["train_clean_tweets.csv", "val_clean_tweets.csv", "test_clean_tweets.csv"]
clean_tweets_all = load_clean_tweets(filenames)

texts_val = tokenize_tweets(tokenizer, clean_tweets_all[clean_tweets_all['type'] == 'val']['cleanTweet'])
texts_train = tokenize_tweets(tokenizer, clean_tweets_all[clean_tweets_all['type'] == 'train']['cleanTweet'])
texts_train_padded = pad_sequences(texts_train, padding='post', maxlen=max_length)
texts_val_padded = pad_sequences(texts_val, padding='post', maxlen=max_length)
y_val = clean_tweets_all[clean_tweets_all['type'] == 'val']['encodedLabel'].values
y_pred = model.predict(texts_val_padded)
y_pred_classes = np.round(y_pred).astype(int).flatten()
print(f'y_val shape: {y_val.shape}, y_pred_classes shape: {y_pred_classes.shape}')

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.show()

def print_classification_report(y_true, y_pred, classes):
    print(classification_report(y_true, y_pred, target_names=classes))

plot_confusion_matrix(y_val, y_pred_classes, classes=['Fake', 'Real'])
print_classification_report(y_val, y_pred_classes, classes=['Fake', 'Real'])
