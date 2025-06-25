import pandas as pd
import numpy as np
import csv
from difflib import SequenceMatcher
import os
import matplotlib.pyplot as plt
import seaborn as sns

def clean_accent(text):
    t = text
    # el
    t = t.replace("Ά", "Α")
    t = t.replace("Έ", "Ε")
    t = t.replace("Ί", "Ι")
    t = t.replace("Ή", "Η")
    t = t.replace("Ύ", "Υ")
    t = t.replace("Ό", "Ο")
    t = t.replace("Ώ", "Ω")
    t = t.replace("ά", "α")
    t = t.replace("έ", "ε")
    t = t.replace("ί", "ι")
    t = t.replace("ή", "η")
    t = t.replace("ύ", "υ")
    t = t.replace("ό", "ο")
    t = t.replace("ώ", "ω")
    t = t.replace("ς", "σ")
    t = t.replace("♡", "")
    t = t.replace("☆", "")
    t = t.replace("*", "")
    return t

def plot_sentiment_distribution(predictions, file_name, results_dir):
    """Plot the distribution of predicted sentiments."""
    unique_sentiments, counts = np.unique(predictions, return_counts=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 10))
    ax = sns.barplot(x=unique_sentiments, y=counts, palette="viridis")
    plt.title("Distribution of Predicted Sentiments", fontsize=20)
    plt.xlabel("Sentiment", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha="center", va="bottom", fontsize=14)

    plot_filename = f"sentiment_plots/{file_name}_sentiment_distribution.png"
    plot_path = f"static/{plot_filename}"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename

def plot_predicted_probabilities(probabilities, results_dir, file_name):
    """Plot the distribution of predicted probabilities."""
    plt.figure(figsize=(16, 10))
    plt.hist(probabilities, bins=50, alpha=0.75, color="blue", label="Predicted probabilities")
    plt.axvline(0.5, color="red", linestyle="dashed", linewidth=2, label="Threshold = 0.5")
    plt.title("Distribution of Predicted Probabilities", fontsize=20)
    plt.xlabel("Predicted Probability", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    
    plot_filename = f"sentiment_plots/{file_name}_predicted_probabilities.png"
    plot_path = f"static/{plot_filename}"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename

def perform_sentiment_analysis(file_name):
    """
    Main function to perform sentiment analysis using the sentistrength algorithm.
    
    Args:
        file_name (str): Name of the input file (without extension)
    
    Returns:
        dict: Dictionary containing analysis results and paths to generated files
    """
    try:
        # Setup paths
        results_dir = f"savedmodel_bin/sentistrength/{file_name}_results"
        os.makedirs(results_dir, exist_ok=True)

        # Load both preprocessed and original data
        print("Loading data...")
        preprocessed_dataset_path = f"files/temp/{file_name}_preprocessed.csv"
        original_dataset_path = f"files/temp/{file_name}.csv"
        
        # Load preprocessed data for prediction
        df_preprocessed = pd.read_csv(preprocessed_dataset_path)
        X_test = df_preprocessed.iloc[:, 0].values  # Get the first column (preprocessed text)

        # Load original data for reference
        df_original = pd.read_csv(original_dataset_path)
        
        # Handle case where original and preprocessed files have different numbers of rows
        # This can happen when preprocessing filters out empty rows
        if len(df_original) != len(df_preprocessed):
            print(f"Warning: Original file has {len(df_original)} rows, preprocessed file has {len(df_preprocessed)} rows")
            print("This is likely due to preprocessing filtering out empty rows.")
            print("Using only the preprocessed data for analysis.")
            
            # Use the preprocessed text as both original and preprocessed
            original_texts = X_test.copy()
        else:
            original_texts = df_original.iloc[:, 0].values  # Get the first column (original text)

        # Load lexicons
        print("Loading lexicons...")
        lexicon_dir = "finallexformysenti"  # Updated path
        
        with open(os.path.join(lexicon_dir, "EmotionLookupTable.txt"), "r", encoding="utf-8") as file:
            terms_list = file.read().splitlines()

        word = []  # arrays for word and score
        score = []
        for t in terms_list:
            t = t.split("\t")
            word.append(t[0])
            score.append(int(t[1]))

        for i in range(len(word)):
            word[i] = clean_accent(word[i].lower())

        # Load emoticons
        with open(os.path.join(lexicon_dir, "EmoticonLookupTable.txt"), "r", encoding="utf-8") as file:
            emotic_list = file.read().splitlines()
        emot = []
        scorem = []
        for te in emotic_list:
            te = te.split("\t")
            emot.append(te[0])
            scorem.append(int(te[1]))

        # Load booster words
        with open(os.path.join(lexicon_dir, "BoosterWordList.txt"), "r", encoding="utf-8") as file:
            terms_listbo = file.read().splitlines()
        boost = []
        scorebo = []
        for tb in terms_listbo:
            tb = tb.split("\t")
            boost.append(tb[0])
            scorebo.append(int(tb[1]))
        for i in range(len(boost)):
            boost[i] = clean_accent(boost[i].lower())

        # Load negating words
        with open(os.path.join(lexicon_dir, "NegatingWordList.txt"), "r", encoding="utf-8") as file:
            terms_listneg = file.read().splitlines()
        neg = []
        for tn in terms_listneg:
            tn = tn.split("\t")
            neg.append(tn[0])
        for i in range(len(neg)):
            neg[i] = clean_accent(neg[i].lower())

        # Constants
        suffix_prune_el = 3
        string_min_score = 0.76
        stikshh = [".", " ", "-", "_", "+", "w", "°", "?", ";", "!", ":", "(", ")"]
        stiksh = [".", " ", "-", "_", "+", "w", "°", "?", ";", "!", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        # Initialize results arrays
        predictions = []
        probabilities = []
        checked_words = 0
        total_words = 0

        print("Making predictions...")
        for review in X_test:
            review = review.replace("\n", "")
            rvwords = review.split(" ")
            review_score = 0
            min_score = -1
            max_score = 1
            flag = False

            for words in rvwords:
                sr = 0
                total_words += 1
                words = clean_accent(words)

                # Check for emoticons
                if words in emot:
                    checked_words += 1
                    sr = scorem[emot.index(words)]
                    review_score += sr
                else:
                    # Handle punctuation
                    a = [""]
                    if "!" in words:
                        a = words.split("!")

                    # Clean unwanted characters
                    for p in range(len(words)):
                        if words[p:p+1] in stikshh:
                            words = words.replace(words[p:p+1], "")
                            words = words.replace(".", "")

                    # Check for negating words
                    if words in neg:
                        checked_words += 1
                        flag = True

                    # Check main lexicon
                    for wrd in [m for m in word if m.lower().startswith(words[:1])]:
                        match = words.find(wrd[:max(3, len(wrd) - suffix_prune_el)])
                        scorera = SequenceMatcher(None, words, wrd).ratio()
                        if match == 0 and scorera > string_min_score:
                            checked_words += 1
                            if flag:
                                flag = False
                            else:
                                sr = score[word.index(wrd)]
                                if a[0] != "":
                                    if sr == -1:
                                        sr = 2
                                    else:
                                        sr += 1
                                review_score += sr

                    # Check booster words
                    if words in boost:
                        checked_words += 1
                        sr = scorebo[boost.index(words)]
                        review_score += sr

                # Update min/max scores
                if sr > max_score:
                    max_score = sr
                if sr < min_score:
                    min_score = sr

            # Calculate final prediction
            sum_min_max = max_score + min_score
            if sum_min_max <= 0:
                prediction = 0
            else:
                prediction = 1

            # Calculate probability (normalized between 0 and 1)
            # First normalize sum_min_max to [-1,1] range
            normalized_sum = max(-1, min(1, sum_min_max))
            probability = (normalized_sum + 1) / 2  # Now this will always be in [0,1] range

            predictions.append(prediction)
            probabilities.append(probability)

        # Convert to numpy arrays
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)

        # Calculate sentiment distribution
        total_texts = len(predictions)
        positive_count = np.sum(predictions == 1)
        negative_count = np.sum(predictions == 0)
        positive_percentage = (positive_count / total_texts) * 100
        negative_percentage = (negative_count / total_texts) * 100

        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'original_text': original_texts,
            'preprocessed_text': X_test,
            'predicted_sentiment': predictions,
            'prediction_probability': probabilities
        })
        
        # Add sentiment labels
        predictions_df['sentiment_label'] = predictions_df['predicted_sentiment'].map({1: 'Positive', 0: 'Negative'})
        
        # Add summary statistics
        if len(predictions_df) > 0:
            predictions_df.loc[0, 'positive_texts_count'] = positive_count
            predictions_df.loc[0, 'negative_texts_count'] = negative_count
            predictions_df.loc[0, 'total_texts'] = total_texts
            predictions_df.loc[0, 'positive_percentage'] = positive_percentage
            predictions_df.loc[0, 'negative_percentage'] = negative_percentage
            predictions_df.loc[0, 'words_found_ratio'] = checked_words / total_words if total_words > 0 else 0

        # Save predictions
        # Create and save predictions with preprocessed text
        preprocessed_predictions_df = predictions_df.copy()
        preprocessed_predictions_df['text'] = preprocessed_predictions_df['preprocessed_text']
        preprocessed_predictions_df = preprocessed_predictions_df.drop(['original_text', 'preprocessed_text'], axis=1)
        preprocessed_output_path = f"{results_dir}/{file_name}_preprocessed_predictions.csv"
        preprocessed_predictions_df.to_csv(preprocessed_output_path, index=False)

        # Create and save predictions with unpreprocessed text
        unpreprocessed_predictions_df = predictions_df.copy()
        unpreprocessed_predictions_df['text'] = unpreprocessed_predictions_df['original_text']
        unpreprocessed_predictions_df = unpreprocessed_predictions_df.drop(['original_text', 'preprocessed_text'], axis=1)
        unpreprocessed_output_path = f"{results_dir}/{file_name}_unpreprocessed_predictions.csv"
        unpreprocessed_predictions_df.to_csv(unpreprocessed_output_path, index=False)

        # Generate plots
        sentiment_dist_path = plot_sentiment_distribution(predictions, file_name, results_dir)
        probability_dist_path = plot_predicted_probabilities(probabilities, results_dir, file_name)

        # Prepare results dictionary
        results = {
            'summary': {
                'positive_count': int(positive_count),
                'negative_count': int(negative_count),
                'total_texts': int(total_texts),
                'positive_percentage': float(positive_percentage),
                'negative_percentage': float(negative_percentage),
                'words_found_ratio': float(checked_words / total_words) if total_words > 0 else 0
            },
            'file_paths': {
                'preprocessed_predictions': preprocessed_output_path,
                'unpreprocessed_predictions': unpreprocessed_output_path,
                'sentiment_distribution': sentiment_dist_path,
                'probability_distribution': probability_dist_path
            }
        }

        return results

    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the sentistrength algorithm for sentiment analysis.")
    parser.add_argument("--file_name", type=str, required=True, help="Name of file to analyze")
    args = parser.parse_args()
    
    results = perform_sentiment_analysis(args.file_name)
    print("\nAnalysis Results:", results)
