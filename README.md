# MATHE Text Analysis Platform

A Flask-based web application for topic modeling, sentiment analysis, and text processing with a focus on Greek language content. This project integrates various machine learning techniques for analyzing journalistic content, including emotional classification and annotation.

## Important Note: System Requirements

**This project requires Linux (Ubuntu recommended) or WSL (Windows Subsystem for Linux) to run properly.** The Hunspell dependency used for text processing is not supported on native Windows systems. Please ensure you are using one of the following:
- Ubuntu 20.04 LTS or higher
- WSL2 with Ubuntu distribution
- Other Linux distributions with Hunspell support

## Features

### Text Analysis
- **Topic Modeling**: 
  - Extract and analyze topics from text data using NMF and LDA
  - Generate word clouds for topic visualization
  - Support for Greek language content
  - Interactive topic exploration

- **Sentiment Analysis**:
  - Real-time sentiment prediction for Greek text
  - Custom sentiment lexicons
  - Neural network-based classification
  - Support for multiple sentiment categories
  - Batch processing capabilities

- **Text Summarization**:
  - Extractive summarization for Greek content
  - Customizable summary length
  - Support for multiple documents
  - Quality metrics for summaries

### File Processing
- CSV file upload and processing
- Support for various text formats
- Batch processing capabilities
- Temporary file management
- Secure file handling

### Data Management
- Dataset organization and storage
- Annotation management
- Results visualization
- Export capabilities

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
│   └── results.html        # Results display
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
sudo apt-get update
sudo apt-get install hunspell
```

3. Install required Python packages:
```bash
pip install -r requirements.txt
```

4. Download required language models:
```bash
python -m spacy download el_core_news_md
```

5. Set up pre-trained models:
   - Copy the trained models from the ML-techniques repository to the appropriate directories in this project
   - Ensure model paths are correctly configured in the application

6. Create required directories:
```bash
mkdir -p data/datasets data/annotations
```

## Usage

### Starting the Application
```bash
python main.py
```
Access the web interface at `http://localhost:5000`

### Main Features

1. **Topic Modeling**
   - Upload text data in CSV format
   - Select number of topics
   - Choose between NMF and LDA algorithms
   - View topic distributions and word clouds
   - Export results

2. **Sentiment Analysis**
   - Input text for real-time analysis
   - Batch process multiple texts
   - View sentiment scores and categories
   - Export analysis results

3. **Text Summarization**
   - Input text for summarization
   - Adjust summary length
   - View and compare summaries
   - Export generated summaries

4. **File Management**
   - Upload and process CSV files
   - Manage temporary files
   - Export processed results
   - View processing history

## API Endpoints

- `/`: Home page
- `/analyze`: Text analysis interface
- `/topic_modelling`: Topic modeling analysis
- `/sentiment`: Sentiment analysis
- `/summarize`: Text summarization
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

3. **Environment variables not loading**
   - Check if .env file exists and is properly formatted
   - Verify environment variables are being loaded in the application

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