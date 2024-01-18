import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
import tensorflow.keras.layers as L
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv1D, GRU
import json

def load_clean_tweets(filenames):
    """
    Load clean tweets from CSV files.
    :param filenames: List of filenames to load.
    :return: Concatenated DataFrame of clean tweets.
    """
    tweet_dfs = [pd.read_csv(filename) for filename in filenames]
    all_clean_tweets = pd.concat(tweet_dfs, ignore_index=True)
    return all_clean_tweets


def initialize_tokenizer(texts):
    """
    Initialize and fit a Keras Tokenizer on given texts.
    :param texts: List or Series of text data.
    :return: Fitted Tokenizer object.
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer


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


filenames = ["train_clean_tweets.csv", "val_clean_tweets.csv", "test_clean_tweets.csv"]
clean_tweets_all = load_clean_tweets(filenames)

# Initiate tokenization
tokenizer = initialize_tokenizer(clean_tweets_all['cleanTweet'])


# Save tokenizer as json file
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

texts_train = tokenize_tweets(tokenizer, clean_tweets_all[clean_tweets_all['type'] == 'train']['cleanTweet'])
texts_val = tokenize_tweets(tokenizer, clean_tweets_all[clean_tweets_all['type'] == 'val']['cleanTweet'])
texts_test = tokenize_tweets(tokenizer, clean_tweets_all[clean_tweets_all['type'] == 'test']['cleanTweet'])
size = len(tokenizer.word_index) + 1

# Labels
y_train = clean_tweets_all[clean_tweets_all['type'] == 'train']['encodedLabel']
y_val = clean_tweets_all[clean_tweets_all['type'] == 'val']['encodedLabel']
y_test = clean_tweets_all[clean_tweets_all['type'] == 'test']['encodedLabel']

# Settings
epoch = 10  
batchSize = 64
outputDimensions = 32
units = 128
learning_rate = 1e-4
l2_reg = 1e-4


# Finding tokenized tweet's max length
max_length = max(max(len(s) for s in texts_train), 
                 max(len(s) for s in texts_val), 
                 max(len(s) for s in texts_test))
with open('max_length.txt', 'w') as f:
    f.write(str(max_length))


tf.keras.backend.clear_session()


texts_train_padded = pad_sequences(texts_train, padding='post', maxlen=max_length)
texts_val_padded = pad_sequences(texts_val, padding='post', maxlen=max_length)

# K-Fold Cross Validation model evaluation
fold_no = 1
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True)

acc_per_fold = []
loss_per_fold = []

for train, val in kfold.split(texts_train_padded, y_train):
  
    # Recreating the model for every fold
    model = tf.keras.Sequential([
        L.Embedding(size, outputDimensions, input_length=max_length),
        Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'),
        GRU(units, return_sequences=True),
        L.GlobalMaxPool1D(),
        L.Dropout(0.3),
        L.Dense(128, activation="relu", kernel_regularizer=l2(l2_reg)),
        L.Dropout(0.3),
        L.Dense(64, activation="relu", kernel_regularizer=l2(l2_reg)),
        L.Dropout(0.3),
        L.Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    
    # Training
    print(f'Training for fold {fold_no} ...')
    modelFit = model.fit(
        texts_train_padded[train], y_train.iloc[train], 
        epochs=epoch,
        validation_data=(texts_train_padded[val], y_train.iloc[val]),
        batch_size=batchSize
    )
    
    # Save the scores
    scores = model.evaluate(texts_train_padded[val], y_train.iloc[val], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    # Increase fold no
    fold_no = fold_no + 1

# Show performance stats.
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

# Function to save the model
def save_model(model, model_name):
    model.save(f"{model_name}.h5")
    print(f"Model saved as {model_name}.h5")

save_model(model, "real_fake_tweets")

# Save model stats.
with open('model_performance.txt', 'w') as f:
    f.write('Score per fold\n')
    for i in range(0, len(acc_per_fold)):
        f.write('------------------------------------------------------------------------\n')
        f.write(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%\n')
    f.write('------------------------------------------------------------------------\n')
    f.write('Average scores for all folds:\n')
    f.write(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})\n')
    f.write(f'> Loss: {np.mean(loss_per_fold)}\n')
    f.write('------------------------------------------------------------------------\n')

