from flask import Flask, request, render_template, send_from_directory, redirect, flash, url_for, jsonify
from datetime import datetime
import random
import os
import logging
import json
import csv_handler
import topic_modelling as tm
import text_processing as tp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = os.urandom(24)

# Create required directories
for directory in ['data/datasets', 'data/annotations', 'files/temp', 'static']:
    os.makedirs(directory, exist_ok=True)

def load_unannotated_texts():
    """Load texts that need annotation from a file"""
    try:
        with open('data/unannotated_texts.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("unannotated_texts.json not found")
        return []

def save_annotated_text(text_id, sentiment, comments):
    """Save annotated text to a file"""
    try:
        # Load all unannotated texts to find the original
        with open('data/unannotated_texts.json', 'r', encoding='utf-8') as f:
            all_texts = json.load(f)
        
        # Find the original text
        original_text = None
        for text in all_texts:
            if text['id'] == text_id:
                original_text = text
                break
        
        if not original_text:
            logger.error(f"Text with ID {text_id} not found")
            return False

        annotation = {
            'text_id': text_id,
            'text': original_text['text'],
            'source': original_text['source'],
            'sentiment': sentiment,
            'comments': comments,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create annotations directory if it doesn't exist
        os.makedirs('data/annotations', exist_ok=True)
        
        # Create or append to the annotations file
        annotations_file = 'data/annotations/annotated_texts.json'
        try:
            # Try to load existing annotations
            with open(annotations_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is empty, start with empty list
            annotations = []
        
        # Add new annotation
        annotations.append(annotation)
        
        # Save all annotations
        with open(annotations_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        logger.error(f"Error saving annotation: {str(e)}")
        return False

@app.route('/')
def home():
    """Home page with feature cards"""
    return render_template('home.html')

@app.route('/analyze')
def analyze():
    """Analysis page with file upload form"""
    return render_template('analyze.html')

@app.route('/analyze', methods=['POST'])
def process_file():
    """Process uploaded file and redirect to analysis"""
    if 'action' not in request.form:
        flash('Invalid form submission', 'error')
        return redirect(url_for('analyze'))

    if request.form["action"] == 'Analyze':
        option = request.form.get('audiofileradio', 'csv')
        logger.info(f"Processing file with option: {option}")

        if option == 'csv':
            if 'audiofile' not in request.files:
                flash('No file part', 'error')
                return redirect(url_for('analyze'))

            uploaded_file = request.files['audiofile']
            if uploaded_file.filename == '':
                flash('No file selected', 'error')
                return redirect(url_for('analyze'))

            if not uploaded_file.filename.endswith('.csv'):
                flash('Only CSV files are allowed', 'error')
                return redirect(url_for('analyze'))

            # Save the uploaded file
            filepath = os.path.join('files/temp', uploaded_file.filename)
            uploaded_file.save(filepath)

            try:
                # Get file preview and column info
                preview = csv_handler.get_preview(filepath)
                column_info = csv_handler.get_column_info(filepath)
                
                # Generate wordcloud
                if tp.generate_wordcloud(filepath):
                    logger.info("Wordcloud generated successfully")
                else:
                    logger.warning("Failed to generate wordcloud")
                
                return render_template(
                    'home_return.html',
                    results_csv=preview,
                    column_info=column_info,
                    filepath=filepath
                )
            except Exception as e:
                logger.error(f"Error processing CSV file: {str(e)}")
                flash('Error processing file. Please try again.', 'error')
                return redirect(url_for('analyze'))
        else:
            flash('Invalid file type selected', 'error')
            return redirect(url_for('analyze'))
    else:
        flash('Invalid action', 'error')
        return redirect(url_for('analyze'))

@app.route('/topic_modelling')
def topic_modelling_form():
    """Display topic modeling form"""
    results_csv = request.args.get('results_csv', 'Unknown results')
    filepath = request.args.get('filepath', 'Unknown path')
    return render_template('home_return.html', results_csv=results_csv, filepath=filepath)

@app.route('/topic_modelling', methods=['POST'])
def topic_modelling_form_results():
    """Process topic modeling form"""
    try:
        filepath = request.args.get('filepath', 'Unknown path')
        number_topics = int(request.form.get('no_topics', 5))
        number_words = int(request.form.get('no_words', 10))
        mode = request.form.get('mode', 'tfidf')

        results_topic_modelling = tm.topic_modelling_function(
            mode, number_topics, number_words, filepath
        )

        return render_template(
            'home_return.html',
            filepath=filepath,
            results_topic_modelling=results_topic_modelling
        )
    except Exception as e:
        logger.error(f"Error in topic modeling: {str(e)}")
        flash('Error performing topic modeling. Please try again.', 'error')
        return redirect(url_for('analyze'))

@app.route('/contribute')
def contribute():
    """Contribution page with dataset upload and sentiment annotation"""
    return render_template('contribute.html')

@app.route('/contribute/dataset', methods=['POST'])
def upload_dataset():
    """Handle dataset upload"""
    if 'dataset' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('contribute'))
    
    file = request.files['dataset']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('contribute'))
    
    if not file.filename.endswith('.csv'):
        flash('Only CSV files are allowed', 'error')
        return redirect(url_for('contribute'))
    
    try:
        # Save the file
        filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join('data/datasets', filename)
        file.save(filepath)
        
        # Save metadata
        metadata = {
            'filename': filename,
            'description': request.form['description'],
            'license': request.form['license'],
            'uploaded_at': datetime.now().isoformat()
        }
        
        with open('data/datasets_metadata.json', 'a', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False)
            f.write('\n')
        
        flash('Dataset uploaded successfully!', 'success')
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        flash('Error uploading dataset. Please try again.', 'error')
    
    return redirect(url_for('contribute'))

@app.route('/api/next-text')
def get_next_text():
    """Get next text for annotation"""
    try:
        # Load all unannotated texts
        with open('data/unannotated_texts.json', 'r', encoding='utf-8') as f:
            all_texts = json.load(f)
        
        # Load already annotated texts
        try:
            with open('data/annotations/annotated_texts.json', 'r', encoding='utf-8') as f:
                annotated_texts = json.load(f)
                annotated_ids = {text['text_id'] for text in annotated_texts}
        except (FileNotFoundError, json.JSONDecodeError):
            annotated_ids = set()
        
        # Filter out already annotated texts
        available_texts = [text for text in all_texts if text['id'] not in annotated_ids]
        
        if not available_texts:
            return jsonify({
                'text': 'No more texts to annotate. Thank you for your contribution!',
                'id': None
            })
        
        # Return a random text from available texts
        text = random.choice(available_texts)
        return jsonify(text)
        
    except Exception as e:
        logger.error(f"Error getting next text: {str(e)}")
        return jsonify({
            'text': 'Error loading text. Please try again.',
            'id': None
        })

@app.route('/contribute/annotate', methods=['POST'])
def submit_annotation():
    """Handle sentiment annotation submission"""
    text_id = request.form.get('text_id')
    sentiment = request.form.get('sentiment')
    comments = request.form.get('comments', '')
    
    if not text_id or not sentiment:
        flash('Please select a sentiment', 'error')
        return redirect(url_for('contribute'))
    
    if save_annotated_text(text_id, sentiment, comments):
        flash('Thank you for your annotation!', 'success')
    else:
        flash('There was an error saving your annotation. Please try again.', 'error')
    
    return redirect(url_for('contribute'))

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Contact page"""
    return render_template('contact.html')

@app.route('/files/<path:filename>')
def download(filename):
    """File download endpoint"""
    return send_from_directory(directory='files', filename=filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
