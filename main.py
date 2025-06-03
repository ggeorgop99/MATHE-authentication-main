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
from flask_session import Session 


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

app.config["SESSION_PERMANENT"] = False # So sessions expire when browser closes (optional)
app.config["SESSION_TYPE"] = "filesystem" # Store sessions on the filesystem
app.config["SESSION_FILE_DIR"] = "./flask_session" # Directory to store session files

Session(app) # Initialize the Session extension

# Create required directories
for directory in ['data/datasets', 'data/annotations', 'files/temp', 'static', app.config["SESSION_FILE_DIR"]]:
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
    logger.info(f"[SESSION_DEBUG] get_or_create_analysis_results called for filepath: '{filepath}'")
    if 'analysis_results' not in session:
        logger.info("[SESSION_DEBUG] 'analysis_results' not in session. Initializing.")
        session['analysis_results'] = {}
        session.modified = True

    if filepath not in session['analysis_results']:
        logger.info(f"[SESSION_DEBUG] filepath '{filepath}' not in session['analysis_results']. Initializing new structure for it.")
        session['analysis_results'][filepath] = {
            'preview': None,
            'preprocessed_preview': None,
            'column_info': None,
            'summary_result': None,
            'topic_modelling_results': None,
            'sentiment_results': None,
            'wordcloud_generated': False,
            'predictions_preview': None
        }
        session.modified = True
    else:
        logger.info(f"[SESSION_DEBUG] filepath '{filepath}' found in session['analysis_results'].")
    
    # Log the current state for this filepath before returning
    # logger.debug(f"[SESSION_DEBUG] Current session data for '{filepath}': {session['analysis_results'].get(filepath)}")
    return session['analysis_results'][filepath]

def store_analysis_results(filepath, results_to_update):
    """Store analysis results in session with proper type conversion"""
    logger.info(f"[SESSION_DEBUG] store_analysis_results called for filepath: '{filepath}'")
    logger.info(f"[SESSION_DEBUG] Results to update: {list(results_to_update.keys())}")

    current_results_for_file = get_or_create_analysis_results(filepath)
    logger.info(f"[SESSION_DEBUG] BEFORE update, session data for '{filepath}': {json.dumps(convert_to_serializable(current_results_for_file.copy()), indent=2)}")

    # Only update the specific keys provided in results_to_update
    for key, value in results_to_update.items():
        if value is not None:  # Only update if the value is not None
            current_results_for_file[key] = convert_to_serializable(value)
    
    logger.info(f"[SESSION_DEBUG] AFTER update, session data for '{filepath}': {json.dumps(convert_to_serializable(current_results_for_file.copy()), indent=2)}")
    session.modified = True
    logger.info(f"[SESSION_DEBUG] session.modified set to True for filepath '{filepath}'.")


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
                        # Get existing analysis results or create new ones
                        analysis_results = get_or_create_analysis_results(filename)
                        
                        # Only reset topic modeling and sentiment analysis results
                        analysis_results['topic_modelling_results'] = None
                        analysis_results['sentiment_results'] = None
                        analysis_results['predictions_preview'] = None
                        session.modified = True
                        
                        # Preprocess the file if not already done
                        if not analysis_results.get('preprocessed_filepath'):
                            preprocessed_filepath = pp.preprocess_file(filepath)
                            analysis_results['preprocessed_filepath'] = preprocessed_filepath
                            session.modified = True
                        else:
                            preprocessed_filepath = analysis_results['preprocessed_filepath']
                        
                        # Generate preview and column info if not already done
                        if not analysis_results.get('preview') or not analysis_results.get('column_info'):
                            preview = csv_handler.get_preview(filepath)
                            preprocessed_preview = csv_handler.get_preview(preprocessed_filepath)
                            column_info = csv_handler.get_column_info(filepath)
                            
                            store_analysis_results(filename, {
                                'preview': preview,
                                'preprocessed_preview': preprocessed_preview,
                                'column_info': column_info,
                                'original_filepath': filepath
                            })
                        
                        # Generate word cloud if not already done
                        if not analysis_results.get('wordcloud_generated'):
                            tp.generate_wordcloud(filepath)
                            store_analysis_results(filename, {'wordcloud_generated': True})
                        
                        # Generate summary if not already done
                        if not analysis_results.get('summary_result'):
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
    filepath = None  # Initialize filepath to None for broader scope in except block if needed
    try:
        filepath = request.form.get('filepath')
        model_name = request.form.get('model_name')
        testing_method = request.form.get('testing_method')
        
        logger.info(f"[SENTIMENT_ROUTE_DEBUG] Received filepath: '{filepath}' for model: {model_name}, method: {testing_method}")

        # Validate uncertainty threshold only for neural models
        if model_name != "sentistrength":
            try:
                uncertainty_threshold = float(request.form.get('uncertainty_threshold', 0.2))
                if testing_method == 'mc':
                    if uncertainty_threshold < 0.05 or uncertainty_threshold > 0.5:
                        flash('Uncertainty threshold must be between 0.05 and 0.5 (5% to 50%)', 'danger')
                        return redirect(url_for('analyze'))
            except ValueError:
                flash('Invalid uncertainty threshold value', 'danger')
                return redirect(url_for('analyze'))

        if not all([filepath, model_name]):
            flash('Missing required parameters', 'danger')
            return redirect(url_for('analyze'))

        # Get current analysis data for preprocessed_filepath check
        current_file_data = get_or_create_analysis_results(filepath)
        
        if 'preprocessed_filepath' not in current_file_data or current_file_data['preprocessed_filepath'] is None:
            original_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
            if os.path.exists(original_upload_path):
                preprocessed_file_actual_path = pp.preprocess_file(original_upload_path)
                store_analysis_results(filepath, {'preprocessed_filepath': preprocessed_file_actual_path})
                current_file_data['preprocessed_filepath'] = preprocessed_file_actual_path
            else:
                flash('Original file not found. Please upload the file again.', 'danger')
                return redirect(url_for('analyze'))
        
        # Ensure 'results_from_sp' and 'predictions_preview_data' are defined before use
        results_from_sp = None
        predictions_preview_data = []

        try:
            # Run sentiment analysis
            results_from_sp = sp.perform_sentiment_analysis(
                file_name=os.path.splitext(filepath)[0],
                model_name=model_name,
                testing_method=testing_method if model_name != "sentistrength" else None,
                uncertainty_threshold=uncertainty_threshold if model_name != "sentistrength" else None
            )
            
            # Read the unpreprocessed predictions file for preview
            if results_from_sp and 'file_paths' in results_from_sp and 'unpreprocessed_predictions' in results_from_sp['file_paths']:
                try:
                    predictions_df = pd.read_csv(results_from_sp['file_paths']['unpreprocessed_predictions'])
                    predictions_preview_data = predictions_df.head(10).to_dict('records')
                except Exception as e_pred_file:
                    logger.error(f"Error reading predictions file: {str(e_pred_file)}")
                    predictions_preview_data = []
            else:
                logger.warning("Predictions file path not found in sentiment analysis results.")

        except Exception as e_sp:
            logger.error(f"Error during sp.perform_sentiment_analysis or predictions_preview creation: {str(e_sp)}", exc_info=True)
            if "InputLayer" in str(e_sp):
                flash('Error: The sentiment model is incompatible with the current TensorFlow version. Please contact the administrator.', 'danger')
            else:
                flash(f'Error performing sentiment analysis: {str(e_sp)}', 'danger')
            return redirect(url_for('analyze'))
            
        # Store the results
        store_analysis_results(filepath, {
            'sentiment_results': results_from_sp,
            'predictions_preview': predictions_preview_data
        })
        
        # Get the complete, updated data for rendering
        analysis_data_for_template = get_or_create_analysis_results(filepath)
        logger.info(f"[SENTIMENT_ROUTE_DEBUG] Final session data for '{filepath}' before rendering: {json.dumps(convert_to_serializable(analysis_data_for_template.copy()), indent=2)}")
        
        return render_template('analysis_results.html',
                               filepath=filepath,
                               results_csv=analysis_data_for_template.get('preview'),
                               preprocessed_csv=analysis_data_for_template.get('preprocessed_preview'),
                               column_info=analysis_data_for_template.get('column_info'),
                               summary_result=analysis_data_for_template.get('summary_result'),
                               sentiment_results=analysis_data_for_template.get('sentiment_results'),
                               results_topic_modelling=analysis_data_for_template.get('topic_modelling_results'),
                               predictions_preview=analysis_data_for_template.get('predictions_preview'),
                               available_models=sp.AVAILABLE_MODELS,
                               testing_methods=sp.TESTING_METHODS,
                               active_tab='sentiment')
    except Exception as e:
        logger.error(f"Overall error in sentiment analysis route (filepath: {filepath}): {str(e)}", exc_info=True)
        flash(f'An unexpected error occurred in sentiment analysis: {str(e)}', 'danger')
        return redirect(url_for('analyze'))

@app.route('/topic_modelling', methods=['POST'])
def topic_modelling_form():
    try:
        # Get basic parameters from form
        no_topics = int(request.form.get('no_topics', 5))
        no_words = int(request.form.get('no_words', 10))
        mode = request.form.get('mode', 'tfidf')
        
        # Get common parameters from form
        max_df = float(request.form.get('max_df', 0.95))
        min_df = int(request.form.get('min_df', 2))
        max_features = int(request.form.get('max_features', 1000))
        max_iter = int(request.form.get('max_iter', 300))
        
        # Get method-specific parameters
        if mode == 'tfidf':
            l1_ratio = float(request.form.get('l1_ratio', 0.5))
            init = request.form.get('init', 'nndsvd')
            params = {
                'l1_ratio': l1_ratio,
                'init': init
            }
        elif mode == 'lda':
            learning_decay = float(request.form.get('learning_decay', 0.7))
            learning_offset = float(request.form.get('learning_offset', 10))
            params = {
                'learning_decay': learning_decay,
                'learning_offset': learning_offset
            }
        elif mode == 'corex':
            anchor_strength = float(request.form.get('anchor_strength', 2.0))
            significance_threshold = float(request.form.get('significance_threshold', 0.05))
            params = {
                'anchor_strength': anchor_strength,
                'significance_threshold': significance_threshold
            }
        else:
            flash('Invalid topic modelling method', 'danger')
            return redirect(url_for('analyze'))
        
        # Validate parameters
        if not (0 <= max_df <= 1):
            flash('Maximum document frequency must be between 0 and 1', 'danger')
            return redirect(url_for('analyze'))
        if min_df < 1:
            flash('Minimum document frequency must be at least 1', 'danger')
            return redirect(url_for('analyze'))
        if max_features < 100:
            flash('Maximum features must be at least 100', 'danger')
            return redirect(url_for('analyze'))
        if max_iter < 100:
            flash('Maximum iterations must be at least 100', 'danger')
            return redirect(url_for('analyze'))
        
        # Method-specific validations
        if mode == 'tfidf':
            if not (0 <= params['l1_ratio'] <= 1):
                flash('L1/L2 ratio must be between 0 and 1', 'danger')
                return redirect(url_for('analyze'))
            if params['init'] not in ['nndsvd', 'random']:
                flash('Invalid initialization method', 'danger')
                return redirect(url_for('analyze'))
        elif mode == 'lda':
            if not (0.5 <= params['learning_decay'] <= 1.0):
                flash('Learning decay must be between 0.5 and 1.0', 'danger')
                return redirect(url_for('analyze'))
            if params['learning_offset'] < 1:
                flash('Learning offset must be at least 1', 'danger')
                return redirect(url_for('analyze'))
        elif mode == 'corex':
            if not (1.0 <= params['anchor_strength'] <= 10.0):
                flash('Anchor strength must be between 1.0 and 10.0', 'danger')
                return redirect(url_for('analyze'))
            if not (0.0 <= params['significance_threshold'] <= 1.0):
                flash('Significance threshold must be between 0.0 and 1.0', 'danger')
                return redirect(url_for('analyze'))
        
        # Get the filepath from the session or request
        filepath = request.form.get('filepath')
        logger.info(f"[TOPIC_ROUTE_DEBUG] Received filepath: '{filepath}'")

        if not filepath:
            flash('No file selected for analysis', 'danger')
            return redirect(url_for('analyze'))
        
        full_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filepath)
        
        # Get existing analysis results (ensures initial structure and marks modified if new)
        analysis_results = get_or_create_analysis_results(filepath) 
        
        # Run topic modeling with all parameters
        results = tm.topic_modelling_function(
            full_filepath, 
            no_topics, 
            no_words, 
            mode,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            max_iter=max_iter,
            **params
        )
        
        # Store ONLY the new topic modeling results. 
        # store_analysis_results will merge this into the existing dictionary.
        store_analysis_results(filepath, {
            'topic_modelling_results': results
        })
        
        # Get the latest results from session to pass to render_template
        analysis_results = get_or_create_analysis_results(filepath) 
        logger.info(f"[TOPIC_ROUTE_DEBUG] Final session data for '{filepath}' before rendering: {json.dumps(convert_to_serializable(analysis_results.copy()), indent=2)}")
        
        # Get predictions preview if it exists (good current logic)
        predictions_preview = []
        if analysis_results.get('sentiment_results') and \
        analysis_results['sentiment_results'].get('file_paths') and \
        analysis_results['sentiment_results']['file_paths'].get('unpreprocessed_predictions'):
            try:
                predictions_df = pd.read_csv(analysis_results['sentiment_results']['file_paths']['unpreprocessed_predictions'])
                predictions_preview = predictions_df.head(10).to_dict('records')
            except Exception as e:
                logger.error(f"Error reading predictions file in topic_modelling_form: {str(e)}")
        
        return render_template('analysis_results.html',
                            filepath=filepath,
                            results_csv=analysis_results.get('preview'),
                            preprocessed_csv=analysis_results.get('preprocessed_preview'),
                            column_info=analysis_results.get('column_info'),
                            results_topic_modelling=analysis_results.get('topic_modelling_results'),
                            summary_result=analysis_results.get('summary_result'),
                            sentiment_results=analysis_results.get('sentiment_results'),
                            predictions_preview=analysis_results.get('predictions_preview'),
                            available_models=sp.AVAILABLE_MODELS,
                            testing_methods=sp.TESTING_METHODS,
                            active_tab='topic')
    
    except Exception as e:
        logger.error(f"Error in topic modelling route: {str(e)}", exc_info=True) # Added exc_info for traceback
        flash(f'Error in topic modelling: {str(e)}', 'danger')
        # It might be better to redirect to the analysis page with the filepath if possible,
        # or a generic analyze page.
        # If filepath is known:
        # return redirect(url_for('analyze_file_results_page', filename=filepath_variable_if_available)) 
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
