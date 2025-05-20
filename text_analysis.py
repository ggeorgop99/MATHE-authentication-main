import pandas as pd
import re
import torch
from transformers import pipeline, AutoTokenizer
import math
from datetime import datetime
import os
from typing import List, Dict, Any

class TextAnalyzer:
    def __init__(self):
        # Initialize with Greek T5 model
        self.model_name = "nlpaueb/grt5-small"
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Load the summarization pipeline
        self.summarizer = pipeline(
            "summarization",
            model=self.model_name,
            tokenizer=self.model_name,
            device=self.device
        )

    def preprocess_greek_tweet(self, text: str) -> str:
        """
        Basic preprocessing for Greek tweets, suitable for transformer models.
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
        # Remove mentions (@username)
        text = re.sub(r"\@\w+", "", text)
        # Remove the hashtag symbol but keep the text
        text = re.sub(r"\#", "", text)
        # Normalize whitespace
        text = " ".join(text.split())
        return text.strip()

    def analyze_text(self, csv_filepath: str) -> Dict[str, Any]:
        """
        Analyze text from a CSV file and return summarization results
        """
        try:
            # Load and preprocess the CSV
            df = pd.read_csv(csv_filepath)
            text_column = df.columns[0]
            
            # Drop rows where the target text column is missing
            df.dropna(subset=[text_column], inplace=True)
            
            # Apply preprocessing
            df["cleaned_text"] = df[text_column].apply(self.preprocess_greek_tweet)
            
            # Filter out empty texts
            df = df[df["cleaned_text"].str.strip() != ""]
            
            if df.empty:
                return {
                    'error': "No valid text data found after preprocessing",
                    'summary': None,
                    'statistics': None,
                    'original_texts': None
                }
            
            # Generate summary
            summary = self.summarize_tweets_transformer_greek(
                df,
                model_name=self.model_name,
                max_chunk_length_words=400,
                summary_max_length=150,
                summary_min_length=30,
                batch_size=8
            )
            
            # Calculate statistics
            stats = {
                'total_documents': len(df),
                'average_length': df['cleaned_text'].str.split().str.len().mean(),
                'summary_length': len(summary.split())
            }
            
            return {
                'summary': summary,
                'statistics': stats,
                'original_texts': df['cleaned_text'].head(5).tolist()
            }
            
        except Exception as e:
            print(f"Error in analyze_text: {str(e)}")
            return {
                'error': str(e),
                'summary': None,
                'statistics': None,
                'original_texts': None
            }

    def summarize_tweets_transformer_greek(
        self,
        df,
        model_name="nlpaueb/grt5-small",
        max_chunk_length_words=400,
        summary_max_length=150,
        summary_min_length=30,
        batch_size=8,
        s1_summary_max_length_factor=1.5
    ) -> str:
        """
        Summarizes Greek text from a DataFrame using a Hugging Face transformer model.
        """
        # Combine all cleaned tweets into a single text block
        all_text_S1 = ". ".join(df["cleaned_text"].astype(str).tolist())
        all_text_S1 = re.sub(r"\.+", ".", all_text_S1)
        all_text_S1 = re.sub(r"\s*\.\s*", ". ", all_text_S1).strip()

        if not all_text_S1:
            return "No text available to summarize after preprocessing."

        # Stage 1: Summarize original tweet chunks
        words_S1 = all_text_S1.split()
        chunks_S1 = [
            " ".join(words_S1[i : i + max_chunk_length_words])
            for i in range(0, len(words_S1), max_chunk_length_words)
        ]

        if not chunks_S1:
            return "Text could not be split into chunks."

        summaries_S1 = []
        num_batches_S1 = math.ceil(len(chunks_S1) / batch_size)
        s1_max_len = int(summary_max_length * s1_summary_max_length_factor)
        s1_min_len = int(summary_min_length * s1_summary_max_length_factor * 0.8)

        for i in range(num_batches_S1):
            batch_chunks_S1 = chunks_S1[i * batch_size : (i + 1) * batch_size]
            try:
                chunk_summaries_S1_result = self.summarizer(
                    batch_chunks_S1,
                    max_length=s1_max_len,
                    min_length=s1_min_len,
                    do_sample=False,
                    truncation=True
                )
                summaries_S1.extend([s["summary_text"] for s in chunk_summaries_S1_result])
            except Exception as batch_error_s1:
                print(f"Error processing Stage 1 batch {i+1}: {batch_error_s1}")
                summaries_S1.extend(["[S1 Error]" for _ in batch_chunks_S1])

        intermediate_long_summary = " ".join(s for s in summaries_S1 if s != "[S1 Error]").strip()

        if not intermediate_long_summary:
            return "No summary generated after Stage 1"

        # Stage 2: Further summarize if needed
        needs_stage2 = (
            len(intermediate_long_summary.split()) > (max_chunk_length_words * 1.1)
            and len(summaries_S1) > 1
        )

        if needs_stage2:
            words_S2_input = intermediate_long_summary.split()
            chunks_S2 = [
                " ".join(words_S2_input[i : i + max_chunk_length_words])
                for i in range(0, len(words_S2_input), max_chunk_length_words)
            ]

            if not chunks_S2:
                return intermediate_long_summary

            summaries_S2 = []
            num_batches_S2 = math.ceil(len(chunks_S2) / batch_size)
            
            for i in range(num_batches_S2):
                batch_chunks_S2 = chunks_S2[i * batch_size : (i + 1) * batch_size]
                try:
                    chunk_summaries_S2_result = self.summarizer(
                        batch_chunks_S2,
                        max_length=summary_max_length,
                        min_length=summary_min_length,
                        do_sample=False,
                        truncation=True
                    )
                    summaries_S2.extend([s["summary_text"] for s in chunk_summaries_S2_result])
                except Exception as batch_error_s2:
                    print(f"Error processing Stage 2 batch {i+1}: {batch_error_s2}")
                    summaries_S2.extend(["[S2 Error]" for _ in batch_chunks_S2])

            final_summary = " ".join(s for s in summaries_S2 if s != "[S2 Error]").strip()
            return final_summary if final_summary else intermediate_long_summary

        return intermediate_long_summary 