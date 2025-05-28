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
import argparse
import os

def mc_dropout_predict(model, x, n_samples=100):
    """
    Perform Monte Carlo dropout predictions to get mean predictions and uncertainty estimates.
    """
    predictions = [model(x, training=True) for _ in range(n_samples)]
    predictions = tf.stack(predictions, axis=0)
    mean_pred = tf.reduce_mean(predictions, axis=0)
    uncertainty = tf.math.reduce_std(predictions, axis=0)
    return mean_pred, uncertainty


def save_plot(fig, filename):
    fig.savefig(filename)
    plt.close(fig)


def plot_sentiment_distribution(predictions, file_name, results_dir):
    """
    Plot the distribution of predicted sentiments.
    """
    unique_sentiments, counts = np.unique(predictions, return_counts=True)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=unique_sentiments, y=counts, palette="viridis")
    plt.title("Distribution of Predicted Sentiments", fontsize=16)
    plt.xlabel("Sentiment", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha="center", va="bottom", fontsize=12)

    plt.savefig(f"{results_dir}/{file_name}_sentiment_distribution.png")
    plt.close()


def plot_predicted_probabilities(probabilities, results_dir):
    """
    Plot the distribution of predicted probabilities.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(probabilities, bins=50, alpha=0.75, color="blue", label="Predicted probabilities")
    plt.axvline(0.5, color="red", linestyle="dashed", linewidth=1, label="Threshold = 0.5")
    plt.title("Distribution of Predicted Probabilities")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{results_dir}/predicted_probabilities.png")
    plt.close()


def plot_uncertainty_distribution(uncertainty, results_dir):
    """
    Plot the distribution of prediction uncertainties.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(uncertainty, bins=50, alpha=0.75, color="purple", label="Prediction uncertainty")
    plt.title("Distribution of Prediction Uncertainties")
    plt.xlabel("Uncertainty")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{results_dir}/uncertainty_distribution.png")
    plt.close()


def plot_predicted_predictions(Y_predictions, results_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(Y_predictions, bins=50, alpha=0.75, color="blue", label="Predictions")
    plt.axvline(
        0.5, color="red", linestyle="dashed", linewidth=1, label="Threshold = 0.5"
    )
    plt.title("Distribution of Predictions")
    plt.xlabel("Prediction")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(f"{results_dir}/predicted_predictions.png")
    plt.show()
    plt.close()

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

def main():
    parser = argparse.ArgumentParser(description="Perform sentiment analysis on unlabeled data.")
    parser.add_argument("--file_name", type=str, required=True, help="Name of the input file")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument(
        "--testing_method",
        type=str,
        required=False,
        choices=["classic", "mc"],
        default="mc",
        help="Testing method (classic or mc dropout)"
    )
    parser.add_argument(
        "--uncertainty_threshold",
        type=float,
        default=0.2,
        help="Threshold for uncertainty in MC dropout predictions"
    )

    args = parser.parse_args()
    file_name = args.file_name
    model_name = args.model_name
    testing_method = args.testing_method
    uncertainty_threshold = args.uncertainty_threshold

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

    # Load and preprocess data
    print("Loading and preprocessing data...")
    dataset_path = f"preprocessed_datasets/{file_name}Spellchecked_bin.csv"
    unpreprocessed_dataset_path = f"datasets/{file_name}_bin.csv"
    
    df = pd.read_csv(dataset_path)
    X_test = df["reviews"].values
    x_test = vec.transform(X_test.astype("U"))

    # Make predictions
    print("Making predictions...")
    if testing_method == "classic":
        Y_probabilities = model.predict(x_test)
        final_predictions = (Y_probabilities > 0.5).astype(int)
        uncertainty = None
    else:  # mc dropout
        mean_pred, uncertainty = mc_dropout_predict(model, x_test.toarray())
        Y_probabilities = mean_pred
        final_predictions = (Y_probabilities > 0.5).numpy().astype(int)
        
        # Identify uncertain predictions
        uncertain_predictions = np.where(uncertainty > uncertainty_threshold)[0]
        print(f"\nUncertainty Analysis:")
        print(f"Total uncertain predictions: {len(uncertain_predictions)}")
        print(f"Uncertain predictions ratio: {len(uncertain_predictions)/len(mean_pred):.2%}")

    # Prepare results
    probs = Y_probabilities.flatten()
    X_test = np.array(X_test).flatten()
    final_predictions = np.array(final_predictions).flatten()
    probs = np.array(probs).flatten()

    # Calculate sentiment distribution
    total_texts = len(X_test)
    positive_count = np.sum(final_predictions == 1)
    negative_count = np.sum(final_predictions == 0)
    positive_percentage = (positive_count / total_texts) * 100
    negative_percentage = (negative_count / total_texts) * 100

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'text': X_test,
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
        uncertainty_flat = uncertainty.numpy().flatten()
        predictions_df['uncertainty'] = uncertainty_flat

    # Save predictions
    output_path = f"{results_dir}/{file_name}_predictions.csv"
    predictions_df.to_csv(output_path, index=False)

    # Create and save predictions with unpreprocessed text
    unpreprocessed_df = pd.read_csv(unpreprocessed_dataset_path)
    unpreprocessed_predictions_df = predictions_df.copy()
    unpreprocessed_predictions_df['text'] = unpreprocessed_df['text'].values
    unpreprocessed_output_path = f"{results_dir}/{file_name}_unpreprocessed_predictions.csv"
    unpreprocessed_predictions_df.to_csv(unpreprocessed_output_path, index=False)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_sentiment_distribution(final_predictions, file_name, results_dir)
    plot_predicted_probabilities(probs, results_dir)
    plot_predicted_predictions(final_predictions, results_dir)
    if testing_method == 'mc':
        plot_uncertainty_distribution(uncertainty_flat, results_dir)

    # Print summary
    print("\nSentiment Analysis Results:")
    print(f"Positive texts: {positive_count} ({positive_percentage:.1f}%)")
    print(f"Negative texts: {negative_count} ({negative_percentage:.1f}%)")
    print(f"Total texts: {total_texts}")
    print(f"\nResults saved to:")
    print(f"- Preprocessed predictions: {output_path}")
    print(f"- Unpreprocessed predictions: {unpreprocessed_output_path}")


if __name__ == "__main__":
    main()
