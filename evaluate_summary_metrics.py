import pandas as pd
import argparse
import os
from collections import Counter

def count_words(text):
    return len(str(text).split())

def redundancy_score(text):
    words = str(text).lower().split()
    if not words:
        return 0.0
    word_counts = Counter(words)
    redundant = sum([count - 1 for count in word_counts.values() if count > 1])
    return redundant / len(words)

def main(csv_path, text_column, summary_txt_path):
    try:
        df = pd.read_csv(csv_path)
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' in CSV file not found.")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    all_text = " ".join(df[text_column].dropna().astype(str).tolist())
    total_words_original = count_words(all_text)

    try:
        with open(summary_txt_path, "r", encoding="utf-8") as f:
            summary_text = f.read()
    except Exception as e:
        print(f"❌ Error opening summary file: {e}")
        return

    total_words_summary = count_words(summary_text)

    if total_words_original == 0:
        print("⚠ No words found in the original text.")
        return

    compression_ratio = total_words_summary / total_words_original
    compression_percent = (1 - compression_ratio) * 100
    redundancy = redundancy_score(summary_text)

    print(f"\n📄 Number of words in original text: {total_words_original}")
    print(f"📝 Number of words in summary: {total_words_summary}")
    print(f"📉 Compression Ratio: {compression_ratio:.3f}")
    print(f"📦 Compression Percentage: {compression_percent:.2f}%")
    print(f"🔁 Redundancy Score: {redundancy:.3f} (ratio of repeated words in the summary)")

    out_log = "evaluation_report_" + os.path.splitext(summary_txt_path)[0] + ".txt"
    with open(out_log, "w", encoding="utf-8") as out:
        out.write(f"CSV file: {csv_path}\n")
        out.write(f"Summary file: {summary_txt_path}\n\n")
        out.write(f"Number of words in original text: {total_words_original}\n")
        out.write(f"Number of words in summary: {total_words_summary}\n")
        out.write(f"Compression Ratio: {compression_ratio:.3f}\n")
        out.write(f"Compression Percentage: {compression_percent:.2f}%\n")
        out.write(f"Redundancy Score: {redundancy:.3f}\n")
    print(f"\n📁 Saved report to: {out_log}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compression & Redundancy metrics for total summary.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file with the tweets.")
    parser.add_argument("--text_column", type=str, default="text", help="Name of the column with the text.")
    parser.add_argument("--summary_txt", type=str, required=True, help="Path to the summary (.txt).")
    args = parser.parse_args()

    main(args.csv_path, args.text_column, args.summary_txt)
