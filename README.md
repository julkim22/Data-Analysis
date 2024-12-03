A Python-based tool for analyzing student text data using clustering techniques and comparative metrics. This tool processes student writing data, performs cluster analysis, and generates visualizations for individual student performance comparisons.

📊 Features
Load and process student text data from CSV files
Calculate text metrics (total words, unique words, ratios)
Perform K-means clustering analysis
Generate comparative visualizations
Individual student performance analysis
Cluster distribution visualization

🔧 Requirements
Copypandas
numpy
scikit-learn
statsmodels
matplotlib

💻 Installation

Clone the repository:

bashCopygit clone https://github.com/yourusername/student-analysis-tool.git

Install required packages:

bashCopypip install pandas numpy scikit-learn statsmodels matplotlib
🚀 Usage
pythonCopy# Import required libraries
from student_analysis import load_data, get_student_data

# Load data
file_path = 'your_data.csv'
student_id = 1738  # Replace with desired student ID

# Run analysis
data = load_data(file_path)
student_data = get_student_data(data, student_id)
📋 Functions
Data Loading and Processing

load_data(file_path): Loads CSV data
get_student_data(data, student_id): Extracts individual student data
calculate_ratios(data): Computes total/unique word ratios
scale_data(data): Standardizes the data
clean_data(data, scaled_data): Removes missing values

Clustering Analysis

perform_kmeans(scaled_data, n_clusters=10): Performs K-means clustering
assign_clusters(data, scaled_data, optimal_k): Assigns cluster labels
plot_elbow(inertia): Visualizes elbow method for optimal cluster selection

Visualization

calculate_comparison(data, student_total_words, student_unique_words, student_total_unique_ratio): Computes comparative metrics
plot_student_vs_group(student_id, student_comparison): Creates comparison visualizations
plot_cluster_distribution(data, student_id): Generates cluster distribution pie chart

📈 Data Format
Required CSV columns:

ID_: Student identifier
Total_words: Total word count
Unique_words: Unique word count

📉 Outputs
The tool generates three types of visualizations:

Elbow plot for optimal cluster selection
Bar charts comparing individual student metrics with group averages
Pie chart showing cluster distribution

⚠️ Limitations

Requires clean CSV data with specified columns
Works with numerical text metrics only
Clustering requires manual selection of optimal K value

🤝 Contributing

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
