import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import spacy
from wordcloud import WordCloud
import re
import logging
import sys
from corextopic import corextopic as ct
import torch

logger = logging.getLogger(__name__)

def clean_text(text):
    """Enhanced text cleaning function"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def topic_modelling_function(csv_filepath, no_topics, no_top_words, mode, 
                           max_df=0.95, min_df=2, max_features=1000,
                           l1_ratio=0.5, max_iter=300, init='nndsvd',
                           learning_decay=0.7, learning_offset=10,
                           anchor_strength=2.0, significance_threshold=0.05):
    """
    Perform topic modelling on the given CSV file.
    
    Args:
        csv_filepath (str): Path to the CSV file
        no_topics (int): Number of topics to extract
        no_top_words (int): Number of top words per topic
        mode (str): Modelling method ('tfidf', 'lda', or 'corex')
        max_df (float): Maximum document frequency (0.0 to 1.0)
        min_df (int): Minimum document frequency
        max_features (int): Maximum number of features
        l1_ratio (float): Mixing parameter for L1/L2 regularization (0.0 to 1.0)
        max_iter (int): Maximum number of iterations
        init (str): Initialization method ('nndsvd' or 'random')
        learning_decay (float): Learning rate decay for LDA (0.5 to 1.0)
        learning_offset (float): Initial learning rate for LDA
        anchor_strength (float): Strength of anchor words for CorEx (1.0 to 10.0)
        significance_threshold (float): Threshold for topic significance in CorEx (0.0 to 1.0)
    """
    try:
        # Read data
        tweets = pd.read_csv(csv_filepath)
        
        # Get the original filename without extension
        original_filename = os.path.splitext(os.path.basename(csv_filepath))[0]

        # Text preprocessing
        tweets['text_processed'] = tweets['text'].astype(str).map(clean_text)
        tweets['text_processed'] = tweets['text_processed'].map(lambda x: re.sub('[,\.!?]', '', x).lower())

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

        # Vectorization
        vectorizer = TfidfVectorizer(
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            stop_words=list(stop_words)
        )
        tfidf = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        # Topic Modeling based on selected method
        if mode == 'tfidf':
            # NMF with TF-IDF
            model = NMF(
                n_components=min(no_topics, int(len(corpus) / 2)),
                random_state=1,
                l1_ratio=l1_ratio,
                max_iter=max_iter,
                init=init
            ).fit(tfidf)
            
            # Get topic words
            topic_words = []
            for topic_idx, topic in enumerate(model.components_):
                words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
                topic_words.append(words)
            
            # Get topic distributions and assignments
            topic_distributions = model.transform(tfidf)
            # Assign topics based on highest probability
            tweets['topic'] = topic_distributions.argmax(axis=1)
            
        elif mode == 'lda':
            # LDA with TF-IDF
            model = LatentDirichletAllocation(
                n_components=min(no_topics, int(len(corpus) / 2)),
                max_iter=max_iter,
                random_state=1,
                learning_decay=learning_decay,
                learning_offset=learning_offset
            ).fit(tfidf)
            
            # Get topic words
            topic_words = []
            for topic_idx, topic in enumerate(model.components_):
                words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
                topic_words.append(words)
            
            # Get topic distributions and assignments
            topic_distributions = model.transform(tfidf)
            # Assign topics based on highest probability
            tweets['topic'] = topic_distributions.argmax(axis=1)
            
        elif mode == 'corex':
            # CorEx
            vectorizer = TfidfVectorizer(
                max_df=0.8,  # Lower max_df to reduce common terms
                min_df=5,    # Increase min_df to focus on more significant terms
                max_features=max_features,
                stop_words=list(stop_words)
            )
            X = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            
            print(f"\nCorEx Debug Info:")
            print(f"Number of documents: {len(corpus)}")
            print(f"Vocabulary size: {len(feature_names)}")
            print(f"Number of topics requested: {no_topics}")
            
            # Train CorEx with explicit parameters
            topic_model = ct.Corex(
                n_hidden=min(no_topics, int(len(corpus) / 2)),
                max_iter=max_iter,
                seed=42,  # Add a fixed seed for reproducibility
                verbose=True  # Add verbose output for debugging
            )
            
            # Fit the model
            topic_model.fit(X, words=feature_names)
            
            # Print topic correlations for debugging
            print("\nTopic Correlations:")
            print(topic_model.tcs)
            
            # Get topic words with proper formatting
            topic_words = []
            used_words = set()  # Track words used in previous topics
            
            for topic_idx in range(topic_model.n_hidden):
                try:
                    # Get topic words and their scores
                    topic_topics = topic_model.get_topics(topic_idx)
                    
                    if not topic_topics:  # If no topics found
                        # Use the most frequent words as fallback
                        word_freq = np.array(X.sum(axis=0)).flatten()
                        top_indices = word_freq.argsort()[-no_top_words:][::-1]
                        words = [feature_names[i] for i in top_indices]
                    else:
                        # Extract words and scores from the topic tuples
                        word_scores = [(word, score) for word, score, _ in topic_topics]
                        # Sort by score
                        word_scores.sort(key=lambda x: x[1], reverse=True)
                        
                        # Select words that haven't been used in previous topics
                        words = []
                        for word, score in word_scores:
                            if word not in used_words and len(words) < no_top_words:
                                words.append(word)
                                used_words.add(word)
                        
                        # If we still need more words, add the most frequent unused words
                        if len(words) < no_top_words:
                            word_freq = np.array(X.sum(axis=0)).flatten()
                            remaining = no_top_words - len(words)
                            # Get indices of words not yet used
                            unused_indices = [i for i, word in enumerate(feature_names) if word not in used_words]
                            if unused_indices:
                                top_indices = sorted(unused_indices, key=lambda i: word_freq[i], reverse=True)[:remaining]
                                additional_words = [feature_names[i] for i in top_indices]
                                words.extend(additional_words)
                                used_words.update(additional_words)
                    
                    topic_words.append(words)
                    print(f"\nTopic {topic_idx + 1} words: {words}")
                    
                except Exception as e:
                    logger.warning(f"Error getting topics for topic {topic_idx}: {str(e)}")
                    # Use the most frequent words as fallback
                    word_freq = np.array(X.sum(axis=0)).flatten()
                    top_indices = word_freq.argsort()[-no_top_words:][::-1]
                    words = [feature_names[i] for i in top_indices]
                    topic_words.append(words)
            
            # Get topic distributions and assignments
            topic_distributions = topic_model.transform(X)
            # Assign topics based on highest probability
            tweets['topic'] = topic_distributions.argmax(axis=1)
            
            # Print topic distribution statistics
            topic_counts = tweets['topic'].value_counts()
            print("\nTopic Distribution:")
            print(topic_counts)
            
            # If no topics were found, raise a more informative error
            if not topic_words:
                raise ValueError(
                    "No topics could be found with the current parameters. Try:\n"
                    "1. Reducing the number of topics (currently {})\n"
                    "2. Increasing the minimum document frequency (currently {})".format(
                        no_topics, min_df
                    )
                )
            
        else:
            raise ValueError(f"Unsupported topic modelling method: {mode}")

        # Save separate CSVs per topic
        output_dir = "output_topics"
        temp_dir = "files/temp"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)

        # Create parameter string for filename
        param_str = f"topics{no_topics}_words{no_top_words}_{mode}"
        if mode == 'tfidf':
            param_str += f"_maxdf{max_df}_mindf{min_df}_maxfeat{max_features}_l1{l1_ratio}_iter{max_iter}_{init}"
        elif mode == 'lda':
            param_str += f"_maxdf{max_df}_mindf{min_df}_maxfeat{max_features}_decay{learning_decay}_offset{learning_offset}_iter{max_iter}"
        elif mode == 'corex':
            param_str += f"_maxdf{max_df}_mindf{min_df}_maxfeat{max_features}_anchor{anchor_strength}_thresh{significance_threshold}_iter{max_iter}"

        for topic_idx in range(no_topics):
            topic_df = tweets[tweets['topic'] == topic_idx]
            # Save in output_topics directory
            topic_filename = os.path.join(output_dir, f"{original_filename}_{param_str}_topic_{topic_idx + 1}.csv")
            topic_df.to_csv(topic_filename, index=False)
            print(f"Saved {topic_filename} with {len(topic_df)} rows")
            
            # Also save in files/temp directory
            temp_topic_filename = os.path.join(temp_dir, f"{original_filename}_{param_str}_topic_{topic_idx + 1}.csv")
            topic_df.to_csv(temp_topic_filename, index=False)
            print(f"Saved {temp_topic_filename} with {len(topic_df)} rows")

        ### Evaluation Metrics ###

        # 1. **Topic Coherence Score**
        tokenized_texts = [doc.split() for doc in corpus]
        dictionary = Dictionary(tokenized_texts)
        corpus_bow = [dictionary.doc2bow(text) for text in tokenized_texts]

        # Convert topic words to the format expected by CoherenceModel
        processed_topic_words = topic_words

        try:
            coherence_model = CoherenceModel(
                topics=processed_topic_words,
                texts=tokenized_texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
        except Exception as e:
            logger.warning(f"Could not calculate coherence score: {str(e)}")
            coherence_score = None

        print(f"Coherence Score: {coherence_score:.4f}" if coherence_score else "Coherence Score: N/A")

        # 2. **Topic Diversity (Unique Words Across Topics)**
        unique_topic_words = set(word for topic in topic_words for word in topic)
        topic_diversity = len(unique_topic_words) / (len(topic_words) * no_top_words)
        print(f"Topic Diversity: {topic_diversity:.4f}")

        # 3. **Histogram of Topic Distribution**
        plt.figure(figsize=(10, 5))
        sns.histplot(tweets['topic'], bins=len(topic_words), kde=True)
        plt.xlabel('Topic')
        plt.ylabel('Document Count')
        plt.title('Topic Distribution Across Documents')
        plt.xticks(range(len(topic_words)), [f"Topic {i + 1}" for i in range(len(topic_words))])
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
            "topic_distribution_plot": "static/topic_distribution.png",
            "parameters": {
                "max_df": max_df,
                "min_df": min_df,
                "max_features": max_features,
                "l1_ratio": l1_ratio,
                "max_iter": max_iter,
                "init": init,
                "mode": mode,
                "learning_decay": learning_decay if mode == 'lda' else None,
                "learning_offset": learning_offset if mode == 'lda' else None,
                "anchor_strength": anchor_strength if mode == 'corex' else None,
                "significance_threshold": significance_threshold if mode == 'corex' else None
            }
        }
    except Exception as e:
        logger.error(f"Error in topic modelling: {str(e)}")
        raise


