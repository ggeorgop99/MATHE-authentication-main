import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import spacy
# import pyLDAvis
# import el_core_news_sm
# import parallelTestModule
#ex PHARM-like


def topic_modelling_function (mode, no_topics, no_top_words, csv_filepath ) :

    # Read data into papers
    # tweets = pd.read_csv('./data/media.csv')
    tweets = pd.read_csv(csv_filepath)

    # Load the regular expression library
    import re
    # Remove punctuation
    tweets['text_processed'] = \
    tweets['text'].map(lambda x: re.sub('[,\.!?]', '', x))
    # Convert the titles to lowercase
    tweets['text_processed'] = \
    tweets['text_processed'].map(lambda x: x.lower())


    def clean_accent(df_col):

        t = df_col

        # el
        t = t.str.replace('Ά', 'Α')
        t = t.str.replace('Έ', 'Ε')
        t = t.str.replace('Ί', 'Ι')
        t = t.str.replace('Ή', 'Η')
        t = t.str.replace('Ύ', 'Υ')
        t = t.str.replace('Ό', 'Ο')
        t = t.str.replace('Ώ', 'Ω')
        t = t.str.replace('ά', 'α')
        t = t.str.replace('έ', 'ε')
        t = t.str.replace('ί', 'ι')
        t = t.str.replace('ή', 'η')
        t = t.str.replace('ύ', 'υ')
        t = t.str.replace('ό', 'ο')
        t = t.str.replace('ώ', 'ω')
        t = t.str.replace('ς', 'σ')

        t = t.str.replace('\n',' ')
        t = t.str.replace('rt', '')

        return t

    def clean_accent_text(text):

        t = text

        # el
        t = t.replace('Ά', 'Α')
        t = t.replace('Έ', 'Ε')
        t = t.replace('Ί', 'Ι')
        t = t.replace('Ή', 'Η')
        t = t.replace('Ύ', 'Υ')
        t = t.replace('Ό', 'Ο')
        t = t.replace('Ώ', 'Ω')
        t = t.replace('ά', 'α')
        t = t.replace('έ', 'ε')
        t = t.replace('ί', 'ι')
        t = t.replace('ή', 'η')
        t = t.replace('ύ', 'υ')
        t = t.replace('ό', 'ο')
        t = t.replace('ώ', 'ω')
        t = t.replace('ς', 'σ')

        t = t.replace('\n',' ')
        t = t.replace('rt', '')

        return t

    tweets['text_processed'] = clean_accent(tweets['text_processed'])
    test_text = tweets['text_processed'].to_numpy()

    nlp = spacy.load('el_core_news_md')




    #topic modeling
    results = []
    results_txt = ''
    mode = mode
    no_topics = no_topics
    no_top_words = no_top_words
    corpus = test_text



    # nlp = spacy.load('el_core_news_md')

    stop_words = nlp.Defaults.stop_words
    stop_words.add('http')

    stop_words_ext = {'https'}
    print (type(stop_words_ext))
    for word in stop_words:
        clean_word = clean_accent_text(word)
        stop_words_ext.add(clean_word)

    stop_words = stop_words.union(stop_words_ext)
    stop_words.add('tco')
    stop_words.add('rt')
    stop_words.add('amp')
    stop_words.add('vmdesignblogg')
    stop_words.add('κι')
    stop_words.add('sleepygeor')
    stop_words.add('mdenaxa')
    stop_words.add('kmitsotakis')
    stop_words.add('kostasvaxevanis')
    stop_words.add('primeministergr')
    stop_words.add('atsipras')
    stop_words.add('androulakisnick')
    stop_words.add('pasok')
    stop_words.add('zefidimadama')
    stop_words.add('knikephoros')
    stop_words.add('of')
    stop_words.add('to')
    stop_words.add('is')
    stop_words.add('renadourou')
    stop_words.add('adieukrinistos_')
    stop_words.add('omada_allagis')
    stop_words.add('avrailas')
    stop_words.add('sotpap1997')
    stop_words.add('odysseas_')
    stop_words.add('velopky')
    stop_words.add('cop27')
    stop_words.add('mononewsgr')
    stop_words.add('businessnewsgr')




    #wordcloud
    from wordcloud import WordCloud
    # Join the different processed titles together.
    long_string = ','.join(list(tweets['text_processed'].values))
    # Create a WordCloud object
    wordcloud = WordCloud(width=1600, height=800, background_color="white", max_words=5000, contour_width=3,
                          contour_color='steelblue', stopwords = stop_words)
    # Generate a word cloud
    wordcloud.generate(long_string)  # Visualize the word cloud
    wordcloud_image = wordcloud.to_image()
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.show()
    # Save the figure to a specific filepath
    filepath = "static/my_wordcloud.png"  # Change this to your desired path
    # plt.savefig(filepath, format="png", dpi=300, bbox_inches="tight")
    wordcloud_image.save(filepath)


    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words=list(stop_words), ngram_range = (1,1))
    tfidf = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    model = NMF(n_components=min(no_topics, int(len(corpus) / 2)), random_state=1, l1_ratio=.5, max_iter=300, init='nndsvd').fit(tfidf)


    #print topics
    for topic_idx, topic in enumerate(model.components_):
        # print("\ttopic %d:" % (topic_idx+1), ', '.join([tfidf_feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        results_txt = results_txt + '('
        for i in topic.argsort()[:-no_top_words - 1:-1]:
            # results.append(tfidf_feature_names[i])
            results_txt = results_txt + feature_names[i] + ', '
        results_txt = results_txt[:-2] + '), '
    results_txt = results_txt[:-2]
    result_string = '\ttopics detected via {}: {}'.format(mode, results_txt)
    print(result_string)
    return result_string


# topic_modelling_function('tdidf',2,4,'./data/media.csv')
