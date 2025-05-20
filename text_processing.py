import pandas as pd
import re
import logging
import os
from wordcloud import WordCloud
import spacy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_accent_text(text):
    """Clean Greek accents from text"""
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

def get_stopwords():
    """Get comprehensive list of stopwords"""
    try:
        nlp = spacy.load('el_core_news_md')
        stop_words = nlp.Defaults.stop_words
        
        # Add common Greek stopwords that are still appearing
        common_greek_words = {
            'και', 'για', 'την', 'που', 'της', 'το', 'τα', 'του', 'των',
            'τον', 'τους', 'τις', 'ο', 'η', 'οι', 'με', 'σε', 'από',
            'προς', 'παρά', 'μετά', 'πριν', 'κατά', 'ανά', 'ενώ', 'όταν', 'εν',
            'εις', 'πως', 'ότι', 'αν', 'ή', 'αλλά', 'όμως', 'ωστόσο',
            'εντούτοις', 'παρόλα', 'παρότι', 'αν και', 'καθώς', 'διότι', 'επειδή',
            'αφού', 'προτού', 'ώστε', 'για να', 'να', 'δε', 'δεν', 'μη', 'μην',
            'όχι', 'ναι', 'μάλιστα', 'βεβαίως', 'σίγουρα', 'ίσως', 'πιθανώς',
            'μάλλον', 'τελικά', 'τελικώς', 'συνολικά', 'συνολικώς', 'σχεδόν',
            'σχετικά', 'περίπου', 'περί', 'γύρω', 'κοντά', 'μακριά', 'πάνω',
            'κάτω', 'μέσα', 'έξω', 'μπροστά', 'πίσω', 'δεξιά', 'αριστερά',
            'εγώ', 'εσύ', 'αυτός', 'αυτή', 'αυτό', 'εμείς', 'εσείς', 'αυτοί', 'αυτές', 'αυτά',
            'μου', 'σου', 'του', 'της', 'μας', 'σας', 'τους', 'στον', 'στην', 'στο',
            'στις', 'στους', 'στα', 'είμαι', 'είσαι', 'είναι', 'είμαστε', 'είστε', 'ήμουν',
            'ήσουν', 'ήταν', 'ήμασταν', 'ήσασταν', 'ήσαν', 'θα', 'έχω', 'έχεις', 'έχει',
            'έχουμε', 'έχετε', 'έχουν', 'κ', 'πχ', 'δηλ', 'κλπ', 'κτλ', 'δηλαδή', 'ακομα',
            'ακομη', 'κανει', 'κανεις', 'πολυ', 'λιγο', 'ολοι', 'ολες', 'ολα', 'πρώτα',
            'πρωτος', 'πρωτη', 'δευτερος', 'δευτερη', 'τριτος', 'τριτη', 'οταν', 'μουσ',
            'μετα', 'ενα', 'ενος', 'μια', 'μιας', 'εναν', 'μιαν', 'αλλες', 'αλλοι',
            'αλλη', 'αλλος', 'αλλο', 'κάποιος', 'κάποια', 'κάποιο', 'τιποτα', 'καθε', 'ιδιο',
            'ιδια', 'ιδιες', 'ιδιοι', 'ιδιων', 'ετσι', 'λοιπον', 'τόσο', 'όσο', 'αυτα', 'αυτες',
            'αυτων', 'αυτους', 'αυτην', 'αυτο', 'εκεινος', 'εκεινη', 'εκείνες', 'εκεινοι', 'εκεινα',
            'μεχρι', 'χωρίς', 'σημερα', 'αυριο', 'χθες', 'τωρα', 'πάντα', 'ποτέ', 'συχνά', 'σπάνια',
            'επειτα', 'τότε', 'πια', 'καν', 'κανενα', 'καμια', 'μας', 'σας', 'πια', 'διοτι',
            'διολου', 'οποιος', 'οποια', 'οποιο', 'ορισμενοι', 'ορισμενες', 'ορισμενα',
            'τετοιος', 'τετοια', 'τετοιο', 'αλλιώς', 'αλλιωτικα', 'αρα', 'αραγε', 'απο', 'ειναι'
            # Add more if specific ones keep appearing
        }
        web_terms = {
            'http', 'https', 'www', 'com', 'org', 'net', 'edu', 'gov', 'gr', 'eu', # added gr, eu
            'tco', 'rt', 'amp', 'html', 'htm', 'php', 'asp', 'jsp', 'xml',
            'url', 'uri', 'domain', 'io', 'co', 'uk', 'news', 'media', 'blog', # common tlds/terms
            'twitter', 'facebook', 'instagram', 'youtube', 'live', 'video', 'photo', 'mononewsgr' # social media terms
        }
        usernames_and_specific_words = { # Renamed for clarity
            'vmdesignblogg', 'κι', 'sleepygeor', 'mdenaxa', # 'κι' might be an actual word, be careful
            'kmitsotakis', 'kostasvaxevanis', 'primeministergr',
            'atsipras', 'androulakisnick', 'pasok', 'zefidimadama',
            'knikephoros', 'of', 'to', 'is', 'renadourou', # English stopwords?
            'adieukrinistos_', 'omada_allagis', 'avrailas',
            'sotpap1997', 'odysseas_', 'velopky', 'cop27', 'cop28', # added cop28
            'mononewsgr', 'businessnewsgr', 'ertnews', 'skai_gr', # common news handles
            'εστια', 'κυβερνηση', 'κυβέρνηση', 'ευρωπαικη', 'ενωση', 'επιτροπη', # common context words to exclude
            'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ', 'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω' # single letters
        }
        
        # Combine all stopwords
        stop_words.update(common_greek_words)
        stop_words.update(web_terms)
        stop_words.update(usernames_and_specific_words)
        
        # Add cleaned versions of all stopwords
        cleaned_stopwords = {clean_accent_text(word) for word in stop_words}
        stop_words.update(cleaned_stopwords)
        
        # Add common patterns that should be removed
        patterns = {
            'http', 'https', 'www', 'tco', 'rt', 'amp',
            'και', 'για', 'την', 'που', 'της', 'το', 'τα',
            'της', 'του', 'των', 'τον', 'την', 'τους', 'τις',
            'ο', 'η', 'οι', 'τα', 'με', 'σε', 'από', 'προς'
        }
        stop_words.update(patterns)
        
        logger.info(f"Total stopwords: {len(stop_words)}")
        return stop_words
    except Exception as e:
        logger.error(f"Error loading stopwords: {str(e)}")
        return set()

def generate_wordcloud(filepath):
    """Generate wordcloud from CSV file"""
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Get the text column (assuming it's the first column)
        text_column = df.columns[0]
        
        # Clean and process the text
        df['text_processed'] = df[text_column].map(lambda x: re.sub('[,\.!?]', '', str(x)))
        df['text_processed'] = df['text_processed'].map(lambda x: x.lower())
        df['text_processed'] = df['text_processed'].map(clean_accent_text)
        
        # Get stopwords
        stop_words = get_stopwords()
        logger.info(f"Using {len(stop_words)} stopwords")
        
        # Additional text cleaning
        def clean_text(text):
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            # Remove common web terms
            text = re.sub(r'\b(tco|rt|amp|www|com|org|net|edu|gov)\b', '', text, flags=re.IGNORECASE)
            # Remove common Greek words
            greek_stop = r'\b(και|για|την|που|της|το|τα|της|του|των|τον|την|τους|τις|ο|η|οι|τα|με|σε|από|προς)\b'
            text = re.sub(greek_stop, '', text, flags=re.IGNORECASE)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # Apply additional cleaning
        df['text_processed'] = df['text_processed'].map(clean_text)
        
        # Join all processed text
        long_string = ' '.join(list(df['text_processed'].values))
        
        # Create and generate wordcloud
        wordcloud = WordCloud(
            width=1600, 
            height=800, 
            background_color="white", 
            max_words=5000, 
            contour_width=3,
            contour_color='steelblue',
            stopwords=stop_words,
            collocations=False,  # Don't include collocations (phrases)
            min_word_length=3,  # Ignore words shorter than 3 characters
            regexp=r"[\w']+",  # Only match word characters and apostrophes
            normalize_plurals=True  # Normalize plurals
        )
        
        # Generate and save the wordcloud
        wordcloud.generate(long_string)
        wordcloud_image = wordcloud.to_image()
        
        # Ensure static directory exists
        os.makedirs('static', exist_ok=True)
        
        # Save with absolute path
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'my_wordcloud.png')
        wordcloud_image.save(save_path)
        logger.info(f"Wordcloud saved to: {save_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error generating wordcloud: {str(e)}")
        return False 