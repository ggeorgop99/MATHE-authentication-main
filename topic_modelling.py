import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import spacy
from wordcloud import WordCloud
import re
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import logging
import sys

logger = logging.getLogger(__name__)

def topic_modelling_function(csv_filepath, no_topics, no_top_words, mode):
    try:
        # Read data
        tweets = pd.read_csv(csv_filepath)

        # Text preprocessing
        tweets['text_processed'] = tweets['text'].astype(str).map(lambda x: re.sub('[,\.!?]', '', x).lower())

        def clean_accent(df_col):
            return df_col.str.replace('Ά', 'Α').str.replace('Έ', 'Ε').str.replace('Ί', 'Ι') \
                .str.replace('Ή', 'Η').str.replace('Ύ', 'Υ').str.replace('Ό', 'Ο') \
                .str.replace('Ώ', 'Ω').str.replace('ά', 'α').str.replace('έ', 'ε') \
                .str.replace('ί', 'ι').str.replace('ή', 'η').str.replace('ύ', 'υ') \
                .str.replace('ό', 'ο').str.replace('ώ', 'ω').str.replace('ς', 'σ') \
                .str.replace('\n', ' ').str.replace('rt', '')

        tweets['text_processed'] = clean_accent(tweets['text_processed'])
        corpus = tweets['text_processed'].to_numpy()

        # Try to load Greek language model
        try:
            nlp = spacy.load('el_core_news_md')
            stop_words = nlp.Defaults.stop_words.union({'http', 'https', 'rt', 'tco', 'amp'})
        except OSError:
            print("\n" + "="*80)
            print("ERROR: Greek language model not found!")
            print("="*80)
            print("\nTo fix this, please run the following command in your terminal:")
            print("\npython -m spacy download el_core_news_md")
            print("\nThis will install the required Greek language model for text processing.")
            print("After installation, please restart the application.")
            print("\n" + "="*80 + "\n")
            raise OSError("Greek language model not installed. Please follow the instructions above to install it.")

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words=list(stop_words))
        tfidf = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        # Topic Modeling with NMF
        model = NMF(n_components=min(no_topics, int(len(corpus) / 2)), random_state=1, l1_ratio=0.5, max_iter=300,
                    init='nndsvd').fit(tfidf)

        # Print topics
        topic_words = []
        for topic_idx, topic in enumerate(model.components_):
            words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
            topic_words.append(words)
            print(f"(Topic {topic_idx + 1}: {', '.join(words)})")

        # Assign each document to a topic
        topic_distributions = model.transform(tfidf)
        tweets['topic'] = topic_distributions.argmax(axis=1)

        # Save separate CSVs per topic
        output_dir = "output_topics"
        os.makedirs(output_dir, exist_ok=True)

        for topic_idx in range(no_topics):
            topic_df = tweets[tweets['topic'] == topic_idx]
            topic_filename = os.path.join(output_dir, f"topic_{topic_idx + 1}.csv")
            topic_df.to_csv(topic_filename, index=False)
            print(f"Saved {topic_filename} with {len(topic_df)} rows")

        ### Evaluation Metrics ###

        # 1. **Topic Coherence Score**
        tokenized_texts = [doc.split() for doc in corpus]
        dictionary = Dictionary(tokenized_texts)
        corpus_bow = [dictionary.doc2bow(text) for text in tokenized_texts]

        coherence_model = CoherenceModel(topics=topic_words, texts=tokenized_texts, dictionary=dictionary, coherence='c_v')

        try:
            coherence_score = coherence_model.get_coherence()
        except RuntimeError:
            coherence_score = None
            print("Coherence score could not be calculated.")

        print(f"Coherence Score: {coherence_score:.4f}" if coherence_score else "Coherence Score: N/A")

        # 2. **Topic Diversity (Unique Words Across Topics)**
        unique_topic_words = set(word for topic in topic_words for word in topic)
        topic_diversity = len(unique_topic_words) / (no_topics * no_top_words)
        print(f"Topic Diversity: {topic_diversity:.4f}")

        # 3. **Histogram of Topic Distribution**
        plt.figure(figsize=(10, 5))
        sns.histplot(tweets['topic'], bins=no_topics, kde=True)
        plt.xlabel('Topic')
        plt.ylabel('Document Count')
        plt.title('Topic Distribution Across Documents')
        plt.xticks(range(no_topics), [f"Topic {i + 1}" for i in range(no_topics)])
        plt.savefig("static/topic_distribution.png")
        print("Saved topic distribution histogram at static/topic_distribution.png")

        # 4. **Show Sample Tweets Per Topic**
        sample_texts = tweets.groupby('topic').apply(lambda df: df.sample(min(len(df), 3)))[['text', 'topic']]
        print("\nTop Example Tweets per Topic:")
        print(sample_texts)

        return {
            "topic_words": topic_words,
            "coherence_score": coherence_score,
            "topic_diversity": topic_diversity,
            "output_dir": output_dir,
            "topic_distribution_plot": "static/topic_distribution.png"
        }
    except Exception as e:
        logger.error(f"Error in topic modelling: {str(e)}")
        raise


