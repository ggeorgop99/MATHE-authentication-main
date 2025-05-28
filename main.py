from flask import Flask, request, render_template, send_from_directory, redirect, flash, url_for, jsonify, session
from datetime import datetime
import random
import os
import logging
import json
import csv_handler
import topic_modelling as tm
import text_processing as tp
import summarization as sum
import sentiment_prediction as sp
import preprocessing as pp
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd

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
app.config['UPLOAD_FOLDER'] = 'files/temp'

# Create required directories
for directory in ['data/datasets', 'data/annotations', 'files/temp', 'static']:
    os.makedirs(directory, exist_ok=True)

def convert_to_serializable(obj):
    """Convert NumPy and pandas types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

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

def get_or_create_analysis_results(filepath):
    """Get analysis results from session or create new ones"""
    if 'analysis_results' not in session:
        session['analysis_results'] = {}
    
    if filepath not in session['analysis_results']:
        session['analysis_results'][filepath] = {
            'preview': None,
            'preprocessed_preview': None,
            'column_info': None,
            'summary_result': None,
            'topic_modelling_results': None,
            'sentiment_results': None,
            'wordcloud_generated': False
        }
    
    return session['analysis_results'][filepath]

def store_analysis_results(filepath, results):
    """Store analysis results in session with proper type conversion"""
    analysis_results = get_or_create_analysis_results(filepath)
    for key, value in results.items():
        analysis_results[key] = convert_to_serializable(value)
    session['analysis_results'][filepath] = analysis_results
    session.modified = True

@app.route('/')
def home():
    """Home page with feature cards"""
    return render_template('home.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        if 'action' in request.form:
            if request.form['action'] == 'Analyze':
                if 'csvfile' not in request.files:
                    flash('No file part', 'danger')
                    return redirect(request.url)
                
                file = request.files['csvfile']
                if file.filename == '':
                    flash('No selected file', 'danger')
                    return redirect(request.url)
                
                if file and file.filename.endswith('.csv'):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    
                    try:
                        # Preprocess the file
                        preprocessed_filepath = pp.preprocess_file(filepath)
                        
                        # Initialize or get analysis results
                        analysis_results = get_or_create_analysis_results(filename)
                        
                        # Generate preview and column info if not already in session
                        if not analysis_results['preview'] or not analysis_results['column_info']:
                            preview = csv_handler.get_preview(filepath)
                            preprocessed_preview = csv_handler.get_preview(preprocessed_filepath)
                            column_info = csv_handler.get_column_info(filepath)
                            
                            # Store all results including preprocessed filepath
                            store_analysis_results(filename, {
                                'preview': preview,
                                'preprocessed_preview': preprocessed_preview,
                                'column_info': column_info,
                                'preprocessed_filepath': preprocessed_filepath,
                                'original_filepath': filepath
                            })
                        
                        # Generate word cloud if not already generated
                        if not analysis_results['wordcloud_generated']:
                            tp.generate_wordcloud(filepath)
                            store_analysis_results(filename, {'wordcloud_generated': True})
                        
                        # Generate summary if not already in session
                        if not analysis_results['summary_result']:
                            summary_result = sum.generate_summary(filepath)
                            store_analysis_results(filename, {'summary_result': summary_result})
                        
                        # Get the latest results from session
                        analysis_results = get_or_create_analysis_results(filename)
                        
                        return render_template('analysis_results.html', 
                                             filepath=filename,
                                             results_csv=analysis_results['preview'],
                                             preprocessed_csv=analysis_results['preprocessed_preview'],
                                             column_info=analysis_results['column_info'],
                                             summary_result=analysis_results['summary_result'],
                                             available_models=sp.AVAILABLE_MODELS,
                                             testing_methods=sp.TESTING_METHODS)
                    except Exception as e:
                        logger.error(f"Error processing file: {str(e)}")
                        flash(f'Error processing file: {str(e)}', 'danger')
                        return redirect(request.url)
                else:
                    flash('Please upload a CSV file', 'danger')
                    return redirect(request.url)
            else:
                flash('Invalid action', 'danger')
                return redirect(request.url)
    
    return render_template('analyze.html')

@app.route('/sentiment_analysis', methods=['POST'])
def sentiment_analysis():
    """Handle sentiment analysis request"""
    try:
        # Get parameters from form
        filepath = request.form.get('filepath')
        model_name = request.form.get('model_name')
        testing_method = request.form.get('testing_method')
        
        # Validate uncertainty threshold
        try:
            uncertainty_threshold = float(request.form.get('uncertainty_threshold', 0.2))
            if testing_method == 'mc':
                if uncertainty_threshold < 0.05 or uncertainty_threshold > 0.5:
                    flash('Uncertainty threshold must be between 0.05 and 0.5 (5% to 50%)', 'danger')
                    return redirect(url_for('analyze'))
        except ValueError:
            flash('Invalid uncertainty threshold value', 'danger')
            return redirect(url_for('analyze'))
        
        if not all([filepath, model_name, testing_method]):
            flash('Missing required parameters', 'danger')
            return redirect(url_for('analyze'))
        
        # Get existing analysis results
        analysis_results = get_or_create_analysis_results(filepath)
        
        # Store current topic modelling results before running sentiment analysis
        current_topic_results = analysis_results.get('topic_modelling_results')
        
        # Check if we have the preprocessed file path
        if 'preprocessed_filepath' not in analysis_results:
            # Try to regenerate the preprocessed file
            original_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
            if os.path.exists(original_filepath):
                preprocessed_filepath = pp.preprocess_file(original_filepath)
                store_analysis_results(filepath, {'preprocessed_filepath': preprocessed_filepath})
            else:
                flash('Original file not found. Please upload the file again.', 'danger')
                return redirect(url_for('analyze'))
        
        # Run sentiment analysis using the preprocessed file
        results = sp.perform_sentiment_analysis(
            file_name=os.path.splitext(filepath)[0],
            model_name=model_name,
            testing_method=testing_method,
            uncertainty_threshold=uncertainty_threshold
        )
        
        # Read the unpreprocessed predictions file for preview
        try:
            predictions_df = pd.read_csv(results['file_paths']['unpreprocessed_predictions'])
            # Get first 10 rows for preview
            predictions_preview = predictions_df.head(10).to_dict('records')
        except Exception as e:
            logger.error(f"Error reading predictions file: {str(e)}")
            predictions_preview = []
        
        # Store results in session, preserving topic modelling results
        store_analysis_results(filepath, {
            'sentiment_results': results,
            'topic_modelling_results': current_topic_results
        })
        
        # Get the latest results from session
        analysis_results = get_or_create_analysis_results(filepath)
        
        return render_template('analysis_results.html',
                             filepath=filepath,
                             results_csv=analysis_results['preview'],
                             preprocessed_csv=analysis_results.get('preprocessed_preview'),
                             column_info=analysis_results['column_info'],
                             summary_result=analysis_results['summary_result'],
                             sentiment_results=results,
                             results_topic_modelling=current_topic_results,
                             predictions_preview=predictions_preview,
                             available_models=sp.AVAILABLE_MODELS,
                             testing_methods=sp.TESTING_METHODS,
                             active_tab='sentiment')
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        flash(f'Error in sentiment analysis: {str(e)}', 'danger')
        return redirect(url_for('analyze'))

@app.route('/topic_modelling', methods=['POST'])
def topic_modelling_form():
    # Get parameters from form
    no_topics = int(request.form.get('no_topics', 5))
    no_words = int(request.form.get('no_words', 10))
    mode = request.form.get('mode', 'tfidf')
    
    # Get the filepath from the session or request
    filepath = request.form.get('filepath')
    if not filepath:
        flash('No file selected for analysis', 'danger')
        return redirect(url_for('analyze'))
    
    full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
    
    # Get existing analysis results
    analysis_results = get_or_create_analysis_results(filepath)
    
    # Run topic modeling
    results = tm.topic_modelling_function(full_filepath, no_topics, no_words, mode)
    
    # Store results in session
    store_analysis_results(filepath, {'topic_modelling_results': results})
    
    # Get the latest results from session
    analysis_results = get_or_create_analysis_results(filepath)
    
    # Get predictions preview if it exists
    predictions_preview = []
    if analysis_results.get('sentiment_results'):
        try:
            predictions_df = pd.read_csv(analysis_results['sentiment_results']['file_paths']['unpreprocessed_predictions'])
            predictions_preview = predictions_df.head(10).to_dict('records')
        except Exception as e:
            logger.error(f"Error reading predictions file: {str(e)}")
    
    return render_template('analysis_results.html',
                         filepath=filepath,
                         results_csv=analysis_results['preview'],
                         preprocessed_csv=analysis_results.get('preprocessed_preview'),
                         column_info=analysis_results['column_info'],
                         results_topic_modelling=results,
                         summary_result=analysis_results['summary_result'],
                         sentiment_results=analysis_results.get('sentiment_results'),
                         predictions_preview=predictions_preview,
                         available_models=sp.AVAILABLE_MODELS,
                         testing_methods=sp.TESTING_METHODS,
                         active_tab='topic')

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

@app.route('/download/<path:filename>')
def download(filename):
    try:
        # Check if the file is from sentiment analysis results
        if 'results' in filename:
            # Extract the directory path from the filename
            directory = os.path.dirname(filename)
            filename = os.path.basename(filename)
            return send_from_directory(directory, filename, as_attachment=True)
        else:
            # For other files, use the default uploads directory
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
