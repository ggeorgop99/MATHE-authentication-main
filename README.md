# iMedius Text Analysis Platform

A Flask-based web application for topic modeling, sentiment analysis, and text processing with a focus on Greek language content. This project integrates various machine learning techniques for analyzing journalistic content, including emotional classification and annotation.

## Important Note: System Requirements

**This project requires Linux (Ubuntu recommended) or WSL (Windows Subsystem for Linux) to run properly.** The Hunspell dependency used for text processing is not supported on native Windows systems. Please ensure you are using one of the following:
- Ubuntu 20.04 LTS or higher
- WSL2 with Ubuntu distribution
- Other Linux distributions with Hunspell support

## Features

### Text Analysis
- **Topic Modeling**: 
  - Extract and analyze topics from text data using NMF, LDA, and CorEx
  - Generate word clouds for topic visualization
  - Support for Greek language content
  - Interactive topic exploration
  - Topic-specific sentiment analysis
  - Topic-specific summarization

- **Sentiment Analysis**:
  - Real-time sentiment prediction for Greek text
  - Multiple model options (Neural Network, SentiStrength)
  - Monte Carlo uncertainty estimation
  - Support for multiple sentiment categories
  - Batch processing capabilities
  - Topic-specific sentiment analysis

- **Text Summarization**:
  - Extractive summarization for Greek content
  - Customizable summary length
  - Support for multiple documents
  - Quality metrics for summaries
  - Topic-specific summarization

### File Processing
- CSV file upload and processing
- Automatic text column standardization
- Support for various text formats
- Batch processing capabilities
- Temporary file management
- Secure file handling
- Caching of preprocessed files for improved performance

### Data Management
- Dataset organization and storage
- Annotation management
- Results visualization
- Export capabilities
- Session-based result storage

## Project Structure

```
MATHE-authentication/
├── main.py                    # Main Flask application
├── topic_modelling.py         # Topic modeling implementation
├── sentiment_prediction.py    # Sentiment analysis implementation
├── summarization.py          # Text summarization implementation
├── preprocessing.py          # Text preprocessing utilities
├── text_processing.py        # General text processing functions
├── csv_handler.py            # CSV file processing utilities
├── sentimark.py              # Sentiment marking utilities
│
├── data/                     # Data directory
│   ├── datasets/            # Training and test datasets
│   └── annotations/         # Annotation files
│
├── static/                   # Static files
│   ├── css/                 # Stylesheets
│   ├── js/                  # JavaScript files
│   └── wordclouds/          # Generated word cloud images
│
├── templates/               # HTML templates
│   ├── base.html           # Base template
│   ├── home.html           # Landing page
│   ├── analyze.html        # Analysis interface
│   └── analysis_results.html # Results display
│
├── savedmodel_bin/         # Saved ML models
├── finallexformysenti/     # Sentiment lexicons
├── output_topics/          # Topic modeling outputs
├── summaries/              # Generated summaries
├── files/                  # Temporary file storage
│   └── temp/              # Temporary files
│
├── stopwords.txt           # Stopwords list
├── stopwords_greek.csv     # Greek stopwords
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Prerequisites

- Python 3.7+
- Flask
- scikit-learn
- spaCy
- pandas
- matplotlib
- WordCloud
- Keras (for neural network models)
- Hunspell (system-level dependency)

## Installation

1. Clone both repositories:
```bash
# Clone this repository
git clone [repository-url]
cd MATHE-authentication

# Clone the model training repository
git clone https://github.com/ggeorgop99/ML-techniques-on-journalistic-content-emotional-classification-and-annotation.git
```

2. Install system dependencies (Linux/WSL only):
```bash
sudo apt update
sudo apt install libhunspell-dev hunspell
sudo apt install -y  autoconf libtool  gettext autopoint
sudo apt install hunspell-el
# now in your env: 
pip install https://github.com/MSeal/cython_hunspell/archive/refs/tags/2.0.3.tar.gz
# gotta get the greek dictionaries from greek hunspell folder and put them in the hunspell dics
# in this location ~/.local/lib/python3.10/site-packages/hunspell/dictionaries
```

3. Install required Python packages:
```bash
pip install -r requirements.txt
```

4. Download required language models and NLTK data:
```bash
# Download spaCy model
python -m spacy download el_core_news_md

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

5. Set up pre-trained models:
   - Copy the trained models from the ML-techniques repository to the appropriate directories in this project
   - Ensure model paths are correctly configured in the application

6. Create required directories:
```bash
mkdir -p data/datasets data/annotations files/temp static/wordclouds output_topics summaries
```

## Usage

### Starting the Application
```bash
python main.py
```
Access the web interface at `http://localhost:5000`

### Main Features

1. **File Upload and Initial Analysis**
   - Upload a CSV file containing text data
   - The system will automatically:
     - Standardize the text column
     - Generate a word cloud
     - Create a text summary
     - Display file preview and column information

2. **Topic Modeling**
   - Select number of topics (default: 5)
   - Choose algorithm (NMF, LDA, or CorEx)
   - Adjust parameters:
     - Maximum document frequency (0-1)
     - Minimum document frequency (≥1)
     - Maximum features (≥100)
     - Maximum iterations (≥100)
     - Algorithm-specific parameters
   - View topic distributions and word clouds
   - Export results

3. **Topic-Specific Analysis**
   - For each topic:
     - Generate a topic-specific summary
     - Perform sentiment analysis on topic content
     - View topic-specific visualizations
     - Export topic-specific results

4. **Sentiment Analysis**
   - Choose model:
     - Neural Network (with Monte Carlo uncertainty)
     - SentiStrength
   - Adjust parameters:
     - Uncertainty threshold (5-50% for Monte Carlo)
   - View sentiment scores and categories
   - Export analysis results

5. **File Management**
   - All processed files are cached for improved performance
   - Temporary files are managed automatically
   - Results can be exported at any time
   - Session-based storage of analysis results

## API Endpoints

- `/`: Home page
- `/analyze`: Text analysis interface
- `/topic_modelling_form`: Topic modeling analysis
- `/topic_summary`: Generate topic-specific summary
- `/topic_sentiment`: Perform topic-specific sentiment analysis
- `/sentiment_analysis`: General sentiment analysis
- `/files/<filename>`: File download
- `/api/next-text`: Get next text for analysis
- `/api/process`: Process text data
- `/api/export`: Export results

## Troubleshooting

### Common Issues

1. **Hunspell not found**
   - Ensure you're running on Linux or WSL
   - Verify Hunspell is installed: `hunspell --version`
   - Install Hunspell if missing: `sudo apt-get install hunspell`

2. **Python package installation issues**
   - Make sure you're using the correct Python version
   - Try updating pip: `pip install --upgrade pip`

3. **File processing errors**
   - Ensure CSV files have a text column named 'text' or 'reviews'
   - Check file encoding (UTF-8 recommended)
   - Verify file permissions in the temp directory

4. **Model loading errors**
   - Verify model files are in the correct directories
   - Check model file permissions
   - Ensure all dependencies are installed

## Contributing

### Code Contribution
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Dataset Contribution
1. Prepare your labeled Greek text dataset in CSV format
2. Visit the Contribute page
3. Upload your dataset with a description and license
4. Your dataset will be used to improve the platform's models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Greek language processing tools and resources
- Open-source ML libraries and frameworks
- Contributors and maintainers of the project
- The [ML-techniques repository](https://github.com/ggeorgop99/ML-techniques-on-journalistic-content-emotional-classification-and-annotation) for providing the trained models

## Contact

For questions and support, please open an issue in the repository or contact [Nikolaos Vryzas](mailto:nvryzas@auth.gr). 