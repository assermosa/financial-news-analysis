import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from src.data_processing.clean_data import clean_data
from src.data_processing.preprocess import clean_text
from src.models.train import build_model
from src.models.evaluate import evaluate_model
from src.models.inference import predict
from src.utils.helpers import setup_logging, save_model
from src.utils.visualization import plot_sentiment_distribution, generate_wordcloud

# Set up logging
setup_logging()

def load_data(filepath):
    """
    Loads the dataset from the given filepath.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """
    Preprocesses the dataset by cleaning and transforming the text.
    """
    # Clean the dataset
    df = clean_data(df)

    # Clean and preprocess text
    df["clean_news"] = df["news"].apply(clean_text)

    # Encode labels
    label_mapping = {"NEGATIVE": 0, "POSITIVE": 1}
    df["label"] = df["sentiment"].map(label_mapping)

    return df

def tokenize_data(df, tokenizer, max_len=128):
    """
    Tokenizes the text data using the BERT tokenizer.
    """
    encoded_data = tokenizer(
        df["clean_news"].tolist(), 
        max_length=max_len, 
        truncation=True, 
        padding='max_length', 
        return_tensors='tf'
    )
    return encoded_data['input_ids'], encoded_data['attention_mask']

def main():
    # Load data
    filepath = '/kaggle/input/news-sentiment-analysis/news.csv'
    df = load_data(filepath)

    # Preprocess data
    df = preprocess_data(df)

    # Tokenize data
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids, attention_mask = tokenize_data(df, tokenizer)

    # Prepare target variables
    y_sentiment = df["label"].values
    y_rank = df["rank_scaled"].values

    # Split data into train and test sets
    X_train, X_test, y_train_sent, y_test_sent, y_train_rank, y_test_rank = train_test_split(
        input_ids.numpy(), y_sentiment, y_rank, test_size=0.2, random_state=42, stratify=y_sentiment
    )
    train_mask, test_mask = train_test_split(
        attention_mask.numpy(), test_size=0.2, random_state=42
    )

    # Build and train the model
    model = build_model()
    history = model.fit(
        [X_train, train_mask], 
        {"sentiment_output": y_train_sent, "rank_output": y_train_rank},
        epochs=8,
        batch_size=64,
        validation_data=([X_test, test_mask], {"sentiment_output": y_test_sent, "rank_output": y_test_rank})
    )

    # Evaluate the model
    eval_results = evaluate_model(model, X_test, test_mask, y_test_sent, y_test_rank)
    print("Evaluation Results:", eval_results)

    # Save the model
    save_model(model, "/kaggle/working/multi_task_bert_model.h5")

    # Visualize results
    plot_sentiment_distribution(df)
    generate_wordcloud(' '.join(df[df["label"] == 1]["clean_news"]), "positive")
    generate_wordcloud(' '.join(df[df["label"] == 0]["clean_news"]), "negative")

    # Example inference
    example_text = "The stock market is experiencing a significant growth due to positive economic indicators."
    sentiment_pred, rank_pred = predict(model, example_text, tokenizer)
    print(f"Sentiment Prediction: {'Positive' if sentiment_pred > 0.5 else 'Negative'}")
    print(f"Rank Prediction: {rank_pred[0][0]}")

if __name__ == "__main__":
    main()