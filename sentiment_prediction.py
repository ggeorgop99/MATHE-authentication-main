import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
import os

# Available models for UI dropdown
AVAILABLE_MODELS = [
    "pharmSpellchecked",
    "pharm_translated_greek_spellchecked",
    "datasetSpellchecked_TL_On_pharmSpellchecked",
    "datasetSpellchecked",
    "datasetAndPharmTranslatedSpellchecked",
    "datasetAndPharmSpellchecked"
]

# Available testing methods for UI dropdown
TESTING_METHODS = ["classic", "mc"]

def mc_dropout_predict(model, x, n_samples=100):
    """
    Perform Monte Carlo dropout predictions to get mean predictions and uncertainty estimates.
    """
    try:
        # Convert sparse matrix to dense if needed
        if hasattr(x, 'toarray'):
            x = x.toarray()
        
        # Ensure x is float32 to match model's expected input
        x = x.astype(np.float32)
        
        # Process in smaller batches to avoid memory issues
        batch_size = 1000
        n_samples = min(n_samples, 50)  # Limit number of samples to avoid memory issues
        
        all_predictions = []
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i + batch_size]
            batch_predictions = []
            
            for _ in range(n_samples):
                # Enable dropout during inference
                batch_pred = model(batch_x, training=True)
                batch_predictions.append(batch_pred)
            
            # Stack predictions for this batch
            batch_predictions = tf.stack(batch_predictions, axis=0)
            all_predictions.append(batch_predictions)
        
        # Combine all batch predictions
        predictions = tf.concat(all_predictions, axis=1)
        
        # Calculate mean and uncertainty
        mean_pred = tf.reduce_mean(predictions, axis=0)
        uncertainty = tf.math.reduce_std(predictions, axis=0)
        
        return mean_pred, uncertainty
        
    except Exception as e:
        print(f"Error in MC dropout prediction: {str(e)}")
        # Fallback to classic prediction if MC dropout fails
        predictions = model(x, training=False)
        return predictions, tf.zeros_like(predictions)


def save_plot(fig, filename):
    fig.savefig(filename)
    plt.close(fig)


def plot_sentiment_distribution(predictions, file_name, results_dir):
    """
    Plot the distribution of predicted sentiments.
    """
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

    # Save plot to static directory
    plot_filename = f"sentiment_plots/{file_name}_sentiment_distribution.png"
    plot_path = f"static/{plot_filename}"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename


def plot_predicted_probabilities(probabilities, results_dir):
    """
    Plot the distribution of predicted probabilities.
    """
    plt.figure(figsize=(16, 10))
    plt.hist(probabilities, bins=50, alpha=0.75, color="blue", label="Predicted probabilities")
    plt.axvline(0.5, color="red", linestyle="dashed", linewidth=2, label="Threshold = 0.5")
    plt.title("Distribution of Predicted Probabilities", fontsize=20)
    plt.xlabel("Predicted Probability", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    
    # Save plot to static directory
    plot_filename = "sentiment_plots/predicted_probabilities.png"
    plot_path = f"static/{plot_filename}"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename


def plot_uncertainty_distribution(uncertainty, results_dir):
    """
    Plot the distribution of prediction uncertainties.
    """
    plt.figure(figsize=(16, 10))
    plt.hist(uncertainty, bins=50, alpha=0.75, color="purple", label="Prediction uncertainty")
    plt.title("Distribution of Prediction Uncertainties", fontsize=20)
    plt.xlabel("Uncertainty", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    
    # Save plot to static directory
    plot_filename = "sentiment_plots/uncertainty_distribution.png"
    plot_path = f"static/{plot_filename}"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename


def plot_predicted_predictions(Y_predictions, results_dir):
    plt.figure(figsize=(16, 10))
    plt.hist(Y_predictions, bins=50, alpha=0.75, color="blue", label="Predictions")
    plt.axvline(0.5, color="red", linestyle="dashed", linewidth=2, label="Threshold = 0.5")
    plt.title("Distribution of Predictions", fontsize=20)
    plt.xlabel("Prediction", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    
    # Save plot to static directory
    plot_filename = "sentiment_plots/predicted_predictions.png"
    plot_path = f"static/{plot_filename}"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename

def calculate_metrics(Y_test, Y_predictions, Y_probabilities, mode):
    if mode == "bin":
        roc_auc = roc_auc_score(Y_test, Y_predictions)
    else:
        roc_auc = roc_auc_score(Y_test, Y_predictions, multi_class="ovr")
    print(f"ROC AUC Score: {roc_auc:.2f}")

    if mode == "bin":
        roc_auc_float = roc_auc_score(Y_test, Y_probabilities)
    else:
        roc_auc_float = roc_auc_score(Y_test, Y_probabilities, multi_class="ovr")
    print(f"ROC AUC Score: {roc_auc_float:.2f}")
    accuracy = np.mean(Y_predictions == Y_test)
    logloss = log_loss(Y_test, Y_probabilities)
    precision = precision_score(
        Y_test, Y_predictions, average="binary" if mode == "bin" else "macro"
    )
    recall = recall_score(
        Y_test, Y_predictions, average="binary" if mode == "bin" else "macro"
    )
    f1 = f1_score(Y_test, Y_predictions, average="binary" if mode == "bin" else "macro")

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    print("\nClassification Report:\n", classification_report(Y_test, Y_predictions))

    metrics_summary = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "ROC AUC",
                "Loss",
            ],
            "Value": [accuracy, precision, recall, f1, roc_auc, logloss],
        }
    )
    print("\nSummary of Evaluation Metrics:\n", metrics_summary)
    return accuracy, logloss, roc_auc, roc_auc_float, recall, precision, f1

def perform_sentiment_analysis(file_name, model_name, testing_method="mc", uncertainty_threshold=0.2):
    """
    Main function to perform sentiment analysis that can be called from the UI.
    
    Args:
        file_name (str): Name of the input file (without extension)
        model_name (str): Name of the model to use (must be one of AVAILABLE_MODELS)
        testing_method (str): Testing method - "classic" or "mc"
        uncertainty_threshold (float): Threshold for uncertainty in MC dropout predictions
    
    Returns:
        dict: Dictionary containing analysis results and paths to generated files
    """
    try:
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Invalid model name. Must be one of: {AVAILABLE_MODELS}")
        
        if testing_method not in TESTING_METHODS:
            raise ValueError(f"Invalid testing method. Must be one of: {TESTING_METHODS}")

        # Setup paths
        dir_path = f"savedmodel_bin/{model_name}_model"
        model_path = f"{dir_path}/{model_name}_bin.keras"
        vectorizer_path = f"{dir_path}/count_vectorizer_{model_name}_bin.pkl"
        results_dir = f"{dir_path}/{file_name}_results"
        os.makedirs(results_dir, exist_ok=True)

        # Load model and vectorizer
        print("Loading model and vectorizer...")
        model = load_model(model_path)
        with open(vectorizer_path, "rb") as f:
            vec = pickle.load(f)

        # Load both preprocessed and original data
        print("Loading data...")
        preprocessed_dataset_path = f"files/temp/{file_name}_preprocessed.csv"
        original_dataset_path = f"files/temp/{file_name}.csv"
        
        # Load preprocessed data for prediction
        df_preprocessed = pd.read_csv(preprocessed_dataset_path)
        X_test = df_preprocessed.iloc[:, 0].values  # Get the first column (preprocessed text)
        x_test = vec.transform(X_test.astype("U"))

        # Load original data for reference
        df_original = pd.read_csv(original_dataset_path)
        original_texts = df_original.iloc[:, 0].values  # Get the first column (original text)

        # Make predictions
        print("Making predictions...")
        if testing_method == "classic":
            Y_probabilities = model.predict(x_test)
            final_predictions = (Y_probabilities > 0.5).astype(int)
            uncertainty = None
        else:  # mc dropout
            mean_pred, uncertainty = mc_dropout_predict(model, x_test)
            Y_probabilities = mean_pred
            final_predictions = (Y_probabilities > 0.5).numpy().astype(int)
            
            # Identify uncertain predictions
            uncertain_predictions = np.where(uncertainty > uncertainty_threshold)[0]
            uncertainty_stats = {
                "total_uncertain": len(uncertain_predictions),
                "uncertain_ratio": len(uncertain_predictions)/len(mean_pred)
            }

        # Prepare results
        probs = Y_probabilities.numpy().flatten() if hasattr(Y_probabilities, 'numpy') else Y_probabilities.flatten()
        X_test = np.array(X_test).flatten()
        original_texts = np.array(original_texts).flatten()
        final_predictions = np.array(final_predictions).flatten()
        probs = np.array(probs).flatten()

        # Calculate sentiment distribution
        total_texts = len(X_test)
        positive_count = np.sum(final_predictions == 1)
        negative_count = np.sum(final_predictions == 0)
        positive_percentage = (positive_count / total_texts) * 100
        negative_percentage = (negative_count / total_texts) * 100

        # Create predictions DataFrame with both original and preprocessed text
        predictions_df = pd.DataFrame({
            'original_text': original_texts,
            'preprocessed_text': X_test,
            'predicted_sentiment': final_predictions,
            'prediction_probability': probs
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

        # Add uncertainty if using MC dropout
        if testing_method == 'mc':
            uncertainty_flat = uncertainty.numpy().flatten() if hasattr(uncertainty, 'numpy') else uncertainty.flatten()
            predictions_df['uncertainty'] = uncertainty_flat

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
        sentiment_dist_path = plot_sentiment_distribution(final_predictions, file_name, results_dir)
        probability_dist_path = plot_predicted_probabilities(probs, results_dir)
        predictions_dist_path = plot_predicted_predictions(final_predictions, results_dir)
        
        if testing_method == 'mc':
            uncertainty_dist_path = plot_uncertainty_distribution(uncertainty_flat, results_dir)

        # Prepare results dictionary
        results = {
            'summary': {
                'positive_count': int(positive_count),
                'negative_count': int(negative_count),
                'total_texts': int(total_texts),
                'positive_percentage': float(positive_percentage),
                'negative_percentage': float(negative_percentage)
            },
            'file_paths': {
                'preprocessed_predictions': preprocessed_output_path,
                'unpreprocessed_predictions': unpreprocessed_output_path,
                'sentiment_distribution': sentiment_dist_path,
                'probability_distribution': probability_dist_path,
                'predictions_distribution': predictions_dist_path
            }
        }

        # Add uncertainty results if using MC dropout
        if testing_method == 'mc':
            results['uncertainty'] = uncertainty_stats
            results['file_paths']['uncertainty_distribution'] = uncertainty_dist_path

        return results

    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        raise  # Re-raise the exception to be handled by the Flask route

if __name__ == "__main__":
    # Example usage
    results = perform_sentiment_analysis(
        file_name="example",
        model_name="pharmSpellchecked",
        testing_method="mc",
        uncertainty_threshold=0.2
    )
    print("\nAnalysis Results:", results)
