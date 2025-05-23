# MATHE Text Analysis Platform

A Flask-based web application for topic modeling, sentiment analysis, and text processing with a focus on Greek language content. This project integrates various machine learning techniques for analyzing journalistic content, including emotional classification and annotation.

## Related Projects

This project works in conjunction with the [ML-techniques-on-journalistic-content-emotional-classification-and-annotation](https://github.com/yourusername/ML-techniques-on-journalistic-content-emotional-classification-and-annotation) repository, which handles:
- Training of sentiment analysis models
- Model development and optimization
- Dataset preparation and preprocessing
- Model evaluation and validation

The models trained in that repository are then used by this web application for real-time sentiment analysis and text processing.

## Features

- **Topic Modeling**: Analyze and extract topics from text data using NMF and LDA
- **Sentiment Analysis**: Process and analyze sentiment in Greek text using pre-trained models
- **Text Summarization**: Generate summaries of long-form content
- **File Processing**: Handle CSV file uploads and processing
- **Content Annotation**: Annotate audio/video files with tampering detection
- **Greek Language Support**: Specialized processing for Greek text content
- **Contribution System**:
  - Upload labeled Greek text datasets
  - Participate in sentiment annotation
  - Help improve model accuracy through crowd-sourcing

## Project Structure

```
MATHE-authentication-main/
├── data/
│   ├── datasets/          # Uploaded CSV datasets
│   └── annotations/       # Annotations data
├── files/
│   └── temp/             # Temporary storage for uploaded files
├── static/
│   ├── css/              # Stylesheets
│   ├── js/               # JavaScript files
│   └── wordclouds/       # Generated word cloud images
├── templates/
│   ├── base.html         # Base template with common elements
│   ├── home.html         # Landing page
│   ├── analyze.html      # File upload and analysis page
│   ├── analysis_results.html  # Results display page
│   ├── about.html        # About page
│   └── contact.html      # Contact page
├── main.py               # Main Flask application
├── csv_handler.py        # CSV file processing utilities
├── text_processing.py    # Text processing and word cloud generation
├── topic_modelling.py    # Topic modeling implementation
└── requirements.txt      # Python dependencies
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

## Installation

1. Clone both repositories:
```bash
# Clone this repository
git clone [repository-url]
cd MATHE-authentication

# Clone the model training repository
git clone https://github.com/yourusername/ML-techniques-on-journalistic-content-emotional-classification-and-annotation.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download required language models:
```bash
python -m spacy download el_core_news_md
```

4. Set up pre-trained models:
   - Copy the trained models from the ML-techniques repository to the appropriate directories in this project
   - Ensure model paths are correctly configured in the application

5. Create required directories:
```bash
mkdir -p data/datasets data/annotations
```

## Usage

1. Start the Flask application:
```bash
python main.py
```

2. Access the web interface at `http://localhost:5000`

3. Main features:
   - Upload CSV files for topic modeling
   - Analyze text sentiment using pre-trained models
   - Generate text summaries
   - Annotate audio/video content
   - View topic modeling results and visualizations
   - Contribute labeled datasets
   - Participate in sentiment annotation

## API Endpoints

- `/`: Home page
- `/analyze`: Text analysis interface
- `/topic_modelling`: Topic modeling analysis
- `/contribute`: Content annotation and dataset contribution interface
- `/about`: About page
- `/contact`: Contact information
- `/files/<filename>`: File download endpoint
- `/api/next-text`: Get next text for sentiment annotation
- `/contribute/dataset`: Upload labeled dataset
- `/contribute/annotate`: Submit sentiment annotation

## ML Components

### Topic Modeling
- Implements both NMF and LDA algorithms
- Supports Greek language processing
- Generates word clouds for visualization

### Sentiment Analysis
- Uses pre-trained models from the [ML-techniques repository](https://github.com/yourusername/ML-techniques-on-journalistic-content-emotional-classification-and-annotation)
- Greek language sentiment analysis
- Custom sentiment lexicons
- Neural network-based classification
- Real-time sentiment prediction
- Crowd-sourced annotation system

### Text Summarization
- Extractive summarization techniques
- Support for Greek language content

## Model Management

The sentiment analysis models used in this application are trained and maintained in a separate repository. To update the models:

1. Train new models in the [ML-techniques repository](https://github.com/yourusername/ML-techniques-on-journalistic-content-emotional-classification-and-annotation)
2. Export the trained models
3. Update the models in this application's model directory
4. Update any model configuration files if necessary

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

### Sentiment Annotation
1. Visit the Contribute page
2. Read the presented Greek text
3. Select the sentiment (Positive/Negative)
4. Add optional comments
5. Submit your annotation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Greek language processing tools and resources
- Open-source ML libraries and frameworks
- Contributors and maintainers of the project
- The [ML-techniques repository](https://github.com/yourusername/ML-techniques-on-journalistic-content-emotional-classification-and-annotation) for providing the trained models

## Contact

For questions and support, please open an issue in the repository or contact [Nikolaos Vryzas](mailto:nvryzas@auth.gr). 