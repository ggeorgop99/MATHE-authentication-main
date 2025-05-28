import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import spacy
from nltk.tokenize import RegexpTokenizer
import numpy as np
import os
from hunspell import Hunspell

# spacy.require_gpu()
nlp = spacy.load("el_core_news_lg")
regexp = RegexpTokenizer("\w+")

# Load stopwords
stopwords = set(
    pd.read_csv("stopwords_greek.csv", header=None).squeeze().tolist()
)

hspell = Hunspell("el_GR")

def spell_check(text):
    corrected_text = []
    for word in text.split():
        if hspell.spell(word):
            corrected_text.append(word)
        else:
            suggestions = hspell.suggest(word)
            corrected_text.append(suggestions[0] if suggestions else word)
    return " ".join(corrected_text)

def lemmatize(text):
    doc = nlp(str(text))
    return " ".join([token.lemma_ for token in doc])

def clean_accent(text):
    accents = {
        "Ά": "Α", "Έ": "Ε", "Ί": "Ι", "Ή": "Η", "Ύ": "Υ", "Ό": "Ο", "Ώ": "Ω",
        "ά": "α", "έ": "ε", "ί": "ι", "ή": "η", "ύ": "υ", "ό": "ο", "ώ": "ω",
        "ς": "σ"
    }
    for accent, char in accents.items():
        text = text.replace(accent, char)
    return text

def preprocess_text(text, stopwords):
    text = clean_accent(text)
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)  # remove urls
    text = re.sub(r"#", "", text)  # remove hashtags
    text = re.sub(r"@\w+", "", text)  # remove mentions
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove numbers
    tokens = regexp.tokenize(text)
    tokens = [token for token in tokens if token not in stopwords and len(token) > 3]
    return " ".join(tokens)

def preprocess_file(file_path):
    """
    Preprocess a CSV file for sentiment analysis.
    If a preprocessed version already exists, it will be returned instead of reprocessing.
    
    Args:
        file_path (str): Path to the input CSV file
    
    Returns:
        str: Path to the preprocessed file
    """
    try:
        # Check if preprocessed file already exists
        input_dir = os.path.dirname(file_path)
        input_filename = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(input_dir, f"{input_filename}_preprocessed.csv")
        
        if os.path.exists(output_path):
            print(f"Preprocessed file already exists at {output_path}")
            return output_path
            
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Get the first text column
        text_column = df.select_dtypes(include=['object']).columns[0]
        
        # Apply preprocessing steps
        df["text"] = df[text_column].apply(spell_check)
        df["text"] = df["text"].apply(lemmatize)
        df["text"] = df["text"].apply(lambda x: preprocess_text(x, stopwords))
        
        # Remove empty rows
        df = df[df["text"].astype(bool)]
        
        # Keep only the preprocessed text column
        df = df[["text"]]
        
        df.dropna(axis=1, how="all")
        
        # Save the preprocessed file
        df.to_csv(output_path, index=False)
        
        return output_path
        
    except Exception as e:
        print(f"Error preprocessing file: {str(e)}")
        raise
