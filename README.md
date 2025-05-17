# MATHE-authentication: Topic Modeling and ML Analysis Platform

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

## Project Structure

```
MATHE-authentication/
├── ML-techniques-on-journalistic-content-emotional-classification-and-annotation/
│   ├── summarization/           # Text summarization models
│   ├── skroutz_scraping/        # Web scraping utilities
│   ├── sentimark/              # Sentiment analysis tools
│   ├── neuralnet/              # Neural network models
│   ├── greek hunspell/         # Greek language processing
│   └── finallexformysenti/     # Sentiment analysis lexicons
├── static/                     # Static files (CSS, JS, images)
├── templates/                  # HTML templates
├── data/                       # Data storage
├── files/                      # Uploaded files
├── main.py                     # Main Flask application
├── topic_modelling.py          # Topic modeling implementation
└── csv_handler.py              # CSV file processing utilities
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

## API Endpoints

- `/`: Home page
- `/topic_modelling`: Topic modeling analysis
- `/contribute`: Content annotation interface
- `/about`: About page
- `/contact`: Contact information
- `/files/<filename>`: File download endpoint

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

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Greek language processing tools and resources
- Open-source ML libraries and frameworks
- Contributors and maintainers of the project
- The [ML-techniques repository](https://github.com/yourusername/ML-techniques-on-journalistic-content-emotional-classification-and-annotation) for providing the trained models

## Contact

For questions and support, please open an issue in the repository or contact the maintainers. 