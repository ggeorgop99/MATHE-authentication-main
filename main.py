from flask import Flask, request, render_template, send_from_directory, redirect, flash, url_for, session, jsonify
from datetime import datetime
import random
import pandas as pd
import os
import csv_handler
import topic_modelling as tm
from text_analysis import TextAnalyzer
import logging
import json

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = os.urandom(24)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize TextAnalyzer
# text_analyzer = TextAnalyzer()

# Store for unannotated texts
UNANNOTATED_TEXTS = []
ANNOTATED_TEXTS = []

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
    # Load the original text
    original_text = None
    for text in UNANNOTATED_TEXTS:
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
    
    # Save to file
    try:
        os.makedirs('data/annotations', exist_ok=True)
        with open('data/annotations/annotated_texts.json', 'a', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False)
            f.write('\n')
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
    if request.form["action"] == 'analysis':
        option = request.form['audiofileradio']
        print(str(option))

        if option == 'csv':
            uploaded_file = request.files['audiofile']
            if uploaded_file.filename != '':
                filepath = "./files/temp/" + uploaded_file.filename
                uploaded_file.save(filepath)
            else:
                filepath = 'unknown'

            if filepath == 'unknown':
                return render_template('analyze.html')

            results_bokeh = filepath
            results_csv = csv_handler.header(filepath)

            print('results')
            print(results_csv)
            print('type')
            print(type(results_csv))

            return render_template(
                'home_return.html',
                results_csv=results_csv,
                results_bokeh=results_bokeh,
                filepath=filepath
            )
        else:
            return 'Thank you for submitting'

@app.route('/topic_modelling')
def topic_modelling_form():
    """Display topic modeling form"""
    results_csv = request.args.get('results_csv', 'Unknown results')
    filepath = request.args.get('filepath', 'Unknown path')
    return render_template('home_return.html', results_csv=results_csv, filepath=filepath)

@app.route('/topic_modelling', methods=['POST'])
def topic_modelling_form_results():
    """Process topic modeling form"""
    results_csv = request.args.get('results_csv', 'Unknown results')
    filepath = request.args.get('filepath', 'Unknown path')
    number_topics = int(request.form.get('no_topics', None))
    number_words = int(request.form.get('no_words', None))
    mode = request.form["mode"]

    print(type(number_topics))
    print(number_words)
    print(mode)
    print(filepath)

    results_topic_modelling = tm.topic_modelling_function(
        mode, number_topics, number_words, filepath
    )

    return render_template(
        'home_return.html',
        results_csv=results_csv,
        filepath=filepath,
        results_topic_modelling=results_topic_modelling
    )

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
    
    # Save the file
    filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join('data/datasets', filename)
    os.makedirs('data/datasets', exist_ok=True)
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
    return redirect(url_for('contribute'))

@app.route('/api/next-text')
def get_next_text():
    """Get next text for annotation"""
    global UNANNOTATED_TEXTS
    
    if not UNANNOTATED_TEXTS:
        UNANNOTATED_TEXTS = load_unannotated_texts()
    
    if not UNANNOTATED_TEXTS:
        return jsonify({
            'text': 'No more texts to annotate. Thank you for your contribution!',
            'id': None
        })
    
    text = UNANNOTATED_TEXTS.pop(0)
    return jsonify(text)

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

@app.route('/results', methods=['GET', 'POST'])
def results():
    """Results page"""
    filename = request.args.get('filename')
    return render_template(filename)

@app.route('/files/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    """File download endpoint"""
    files = "./files/"
    return send_from_directory(directory=files, filename=filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
