import pandas as pd
import re
import html
import string
from nltk.corpus import stopwords

# Constants
punctuations = string.punctuation
stops = stopwords.words("english")

def load_data(file_path):
    """
    Loads data from a CSV file and returns a DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def check_null_values(df, set_name):
    """
    Checks for null values in the DataFrame.
    """
    try:
        print(f"{set_name} Set:\n", df.isnull().any())
    except AttributeError:
        print("Invalid DataFrame passed for null value check.")

def clean_tweets(tweets):
    """
    Cleans a list of tweets.
    """
    cleaned_tweets = []
    for tweet in tweets:
        tweet = html.unescape(tweet)
        tweet = re.sub(r"@\w+", " ", tweet)
        tweet = re.sub(r"http\S+", " ", tweet)
        tweet = "".join([char for char in tweet if char not in punctuations])
        tweet = tweet.lower()
        tweet_words = tweet.split()
        cleaned_tweet = " ".join([word for word in tweet_words if word not in stops])
        cleaned_tweets.append(cleaned_tweet)
    return cleaned_tweets

def process_data(df):
    """
    Processes the data by cleaning tweets and handling label encoding.
    """
    if df is not None:
        df["cleanTweet"] = clean_tweets(df["tweet"].copy())
        if "label" in df.columns:
            df["encodedLabel"] = pd.get_dummies(df["label"])["real"]
            df['encodedLabel'] = df['encodedLabel'].astype(int)

def concatenate_clean_tweets(train_df, val_df, test_df):
    """
    Concatenates clean tweets from training, validation, and testing datasets.
    """
    train_clean = train_df["cleanTweet"].copy() if "cleanTweet" in train_df else pd.Series([])
    valid_clean = val_df["cleanTweet"].copy() if "cleanTweet" in val_df else pd.Series([])
    test_clean = test_df["cleanTweet"].copy() if "cleanTweet" in test_df else pd.Series([])

    all_clean_tweets = pd.concat([train_clean, valid_clean, test_clean], ignore_index=True)
    return all_clean_tweets

# Main script execution
if __name__ == "__main__":
    # Load data
    train_df = load_data("Constraint_Train.csv")
    print(train_df.columns)
    val_df = load_data("Constraint_Val.csv")
    print(val_df.columns)
    test_df = load_data("Constraint_Test.csv")
    print(test_df.columns)

    # Check for null values
    check_null_values(train_df, "Training")
    check_null_values(val_df, "Validation")
    check_null_values(test_df, "Testing")

    # Process data
    process_data(train_df)
    process_data(val_df)
    process_data(test_df)
    all_clean_tweets = concatenate_clean_tweets(train_df, val_df, test_df)

    # Check the result
    print(all_clean_tweets)
    print(len(all_clean_tweets))

    # Print first few rows of the dataframes
    print(train_df.head())
    print(val_df.head())
    print(test_df.head())

    train_df['type'] = 'train'
    val_df['type'] = 'val'  
    test_df['type'] = 'test'
    train_df[['cleanTweet', 'type', 'encodedLabel']].to_csv("train_clean_tweets.csv", index=False)
    val_df[['cleanTweet', 'type', 'encodedLabel']].to_csv("val_clean_tweets.csv", index=False)
    #test_df[['cleanTweet', 'type', 'encodedLabel']].to_csv("test_clean_tweets.csv", index=False)


