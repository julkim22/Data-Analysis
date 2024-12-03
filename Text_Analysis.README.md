A Python-based tool for analyzing and comparing linguistic features in text data. This tool performs natural language processing to extract various metrics from texts and creates comparative visualizations.
📊 Features

Text Analysis: Sentence tokenization and comprehensive linguistic analysis
POS Tagging: Detailed part of speech analysis
Metrics Calculation: Various linguistic metrics including sentence complexity
Visualization: Comparative bar charts for multiple texts
CSV Support: Process data directly from CSV files

🚀 Getting Started
Prerequisites
bashCopypython 3.x
nltk
matplotlib
numpy
pandas
seaborn
Installation

Clone the repository:
bashCopygit clone https://github.com/yourusername/text-analysis-tool.git

Install required packages:
bashCopypip install nltk matplotlib numpy pandas seaborn

Download NLTK data:
pythonCopypython -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"


💻 Usage
pythonCopyimport pandas as pd
from text_analyzer import analyze_text, visualize_comparison

# Load your data
data = pd.read_csv('your_file.csv')
text = data['transcript'].dropna().tolist()

# Analyze texts
analysis_results1 = analyze_text(text[0])  # First text
analysis_results2 = analyze_text(text[-1])  # Last text

# Visualize comparison
visualize_comparison([analysis_results1, analysis_results2], ["Text 1", "Text 2"])
📈 Analysis Metrics
MetricDescriptionAverage Sentence LengthMean number of words per sentenceWord TypesCounts of nouns, verbs, adjectives, etc.Sentence ComplexityDistribution of simple, compound, and complex sentences
Sentence Classification:

Simple: ≤10 words
Compound: 11-20 words
Complex: >20 words

🛠️ Functions
analyze_text(text)
Performs linguistic analysis on input text.
Parameters:

text (str): Input text to analyze

Returns:

Dictionary with calculated metrics

visualize_comparison(analysis_results_list, labels)
Creates comparative visualization.
Parameters:

analysis_results_list (list): Analysis results dictionaries
labels (list): Text labels for visualization

⚠️ Limitations

English language texts only
Minimum two texts required for comparison
Processing time depends on text size

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🤝 Contributing

Fork the project
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit changes (git commit -m 'Add AmazingFeature')
Push to branch (git push origin feature/AmazingFeature)
Open a Pull Request
