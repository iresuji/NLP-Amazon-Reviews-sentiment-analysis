import spacy
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from transformers import pipeline

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample function for sentiment analysis
def analyze_sentiment(text):
    doc = nlp(text)
    
    # Extract sentiment score from spaCy
    sentiment_score = doc.sentiment

    if sentiment_score >= 0.5:
        return "Positive"
    elif sentiment_score <= -0.5:
        return "Negative"
    else:
        return "Neutral"

# Function for text preprocessing
def preprocess_text(text):
    # Check if text is NaN (float type), convert to empty string if true
    if pd.isna(text):
        return ""
    
    # Remove stop words
    tokens = [token.text for token in nlp(text) if token.text.lower() not in STOP_WORDS]
    
    # Other preprocessing steps as needed
    
    return " ".join(tokens)

# Load dataset
data = pd.read_csv("amazon_product_reviews.csv")

# Apply text preprocessing to the 'reviews.text' column
data['cleaned_text'] = data['reviews.text'].apply(preprocess_text)

def sentiment_analysis_model(text):
   # Load pre-trained sentiment analysis model
    sentiment_pipeline = pipeline('sentiment-analysis', device=0)

    # Get the sentiment prediction for the input text
    result = sentiment_pipeline(text)[0]

    # Return the predicted sentiment
    return result['label']

# Test the sentiment analysis model on sample reviews
for idx, review_text in enumerate(data['reviews.text'].head(5)):
    cleaned_text = data['cleaned_text'].iloc[idx]
    predicted_sentiment = sentiment_analysis_model(cleaned_text)
    print(f"Original Review {idx + 1}: {review_text}")
    print(f"Cleaned Review {idx + 1}: {cleaned_text}")
    print(f"Predicted Sentiment {idx + 1}: {predicted_sentiment}\n")
