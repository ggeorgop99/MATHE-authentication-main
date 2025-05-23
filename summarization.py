import pandas as pd
import re
import torch
from transformers import pipeline, AutoTokenizer
import math
from datetime import datetime
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_greek_tweet(text):
    """Basic preprocessing for Greek tweets."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+", "", text)
    text = re.sub(r"\#", "", text)
    text = " ".join(text.split())
    return text.strip()

def load_and_preprocess_greek(csv_path, text_column="text"):
    """Loads and preprocesses Greek text from CSV."""
    try:
        df = pd.read_csv(csv_path)
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV file.")
        
        df.dropna(subset=[text_column], inplace=True)
        df["cleaned_text"] = df[text_column].apply(preprocess_greek_tweet)
        df = df[df["cleaned_text"].str.strip() != ""]
        
        return df
    except Exception as e:
        logger.error(f"Error in data loading/preprocessing: {e}")
        return None

def generate_summary(csv_path, text_column="text", model_name="IMISLab/GreekT5-mt5-small-greeksum"):
    """
    Main function to generate summary from CSV file using two-stage summarization.
    Returns a dictionary containing the summary and any error messages.
    """
    try:
        # Load and preprocess data
        df = load_and_preprocess_greek(csv_path, text_column)
        if df is None or df.empty:
            return {"error": "No valid text data found after preprocessing."}

        # Combine all cleaned tweets
        all_text = ". ".join(df["cleaned_text"].astype(str).tolist())
        all_text = re.sub(r"\.+", ".", all_text)
        all_text = re.sub(r"\s*\.\s*", ". ", all_text).strip()

        if not all_text:
            return {"error": "No text available to summarize after preprocessing."}

        # Initialize summarizer
        try:
            logger.info(f"Initializing summarization pipeline with model: {model_name}")
            device = 0 if torch.cuda.is_available() else -1
            if device == 0:
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("No GPU detected, using CPU (summarization might be slow).")

            # Use the simpler pipeline initialization that works in new_bart_sum.py
            summarizer = pipeline(
                "summarization",
                model=model_name,
                tokenizer=model_name,
                device=device
            )
            logger.info("Pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize summarizer: {str(e)}")
            return {"error": f"Failed to initialize summarizer: {str(e)}"}

        # Stage 1: Initial summarization of chunks
        max_chunk_length_words = 400
        words = all_text.split()
        chunks = [
            " ".join(words[i : i + max_chunk_length_words])
            for i in range(0, len(words), max_chunk_length_words)
        ]

        if not chunks:
            return {"error": "Text could not be split into chunks."}

        # Process chunks in batches
        batch_size = 8
        summaries_S1 = []
        num_batches = math.ceil(len(chunks) / batch_size)
        s1_max_len = 225  # 150 * 1.5
        s1_min_len = 45   # 30 * 1.5

        for i in range(num_batches):
            batch_chunks = chunks[i * batch_size : (i + 1) * batch_size]
            try:
                chunk_summaries = summarizer(
                    batch_chunks,
                    max_length=s1_max_len,
                    min_length=s1_min_len,
                    do_sample=False,
                    truncation=True
                )
                summaries_S1.extend([s["summary_text"] for s in chunk_summaries])
            except Exception as batch_error:
                logger.error(f"Error processing batch {i+1}: {batch_error}. Skipping this batch.")
                summaries_S1.extend(["[S1 Error]" for _ in batch_chunks])

        intermediate_summary = " ".join(s for s in summaries_S1 if s != "[S1 Error]").strip()

        if not intermediate_summary:
            return {"error": "No summary generated after Stage 1."}

        # Stage 2: Final summarization if needed
        needs_stage2 = (
            len(intermediate_summary.split()) > (max_chunk_length_words * 1.1)
            and len(summaries_S1) > 1
        )

        if needs_stage2:
            words_S2 = intermediate_summary.split()
            chunks_S2 = [
                " ".join(words_S2[i : i + max_chunk_length_words])
                for i in range(0, len(words_S2), max_chunk_length_words)
            ]

            if not chunks_S2:
                return {
                    "summary": intermediate_summary,
                    "num_texts": len(df),
                    "model_used": model_name,
                    "stages": 1
                }

            summaries_S2 = []
            num_batches_S2 = math.ceil(len(chunks_S2) / batch_size)

            for i in range(num_batches_S2):
                batch_chunks_S2 = chunks_S2[i * batch_size : (i + 1) * batch_size]
                try:
                    chunk_summaries_S2 = summarizer(
                        batch_chunks_S2,
                        max_length=150,
                        min_length=30,
                        do_sample=False,
                        truncation=True
                    )
                    summaries_S2.extend([s["summary_text"] for s in chunk_summaries_S2])
                except Exception as batch_error_s2:
                    logger.error(f"Error processing Stage 2 batch {i+1}: {batch_error_s2}. Skipping this batch.")
                    summaries_S2.extend(["[S2 Error]" for _ in batch_chunks_S2])

            final_summary = " ".join(s for s in summaries_S2 if s != "[S2 Error]").strip()
            
            if final_summary:
                return {
                    "summary": final_summary,
                    "num_texts": len(df),
                    "model_used": model_name,
                    "stages": 2
                }
            else:
                return {
                    "summary": intermediate_summary,
                    "num_texts": len(df),
                    "model_used": model_name,
                    "stages": 1
                }
        else:
            return {
                "summary": intermediate_summary,
                "num_texts": len(df),
                "model_used": model_name,
                "stages": 1
            }

    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return {"error": f"Error generating summary: {str(e)}"} 