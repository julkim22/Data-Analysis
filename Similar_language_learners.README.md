A Python application that uses Pinecone vector database to find similar language learners based on their total words, unique words, age, and gender.

🚀 Features
Vector similarity search using Pinecone
Student data processing and vectorization
Filtered search based on age and gender
Metadata storage for comprehensive user information
Serverless deployment on AWS

📋 Prerequisites
Python 3.x
Pinecone API key
AWS account (for serverless deployment)

Required Python packages:
pinecone-client
pandas



🔧 Installation
Install required packages:
bashCopypip install pinecone-client pandas
Set up your Pinecone API key:
pythonCopyPINECONE_API_KEY = "your-api-key-here"

💻 Usage
Initialize Pinecone Index
pythonCopyfrom pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "language"

# Create index
pc.create_index(
    name=index_name,
    dimension=2,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)
Load and Process Data
pythonCopyimport pandas as pd

# Load data
data = pd.read_csv('your_data.csv')
data = data.dropna(subset=['Total_words', 'Unique_words'])
Find Similar Users
pythonCopy# Find similar users based on metrics and demographics
results = get_similar_user(
    total_words=223,
    unique_words=115,
    age=8,
    gender="남"
)
📊 Data Structure
Vector Format

Dimension: 2
Features:

Total words
Unique words



Metadata Fields

pastEnglishLearningMethodChild
PresentEnglishLearningMethodChild
grade
year_birth
game_experience
gender
device
group
lang_level
age

🔍 Functions
get_similar_user(total_words, unique_words, age, gender)
Finds similar users based on language metrics and demographics.
Parameters:

total_words: Total word count
unique_words: Unique word count
age: User's age
gender: User's gender

Returns:

Top 3 most similar users with their metadata

⚙️ Configuration

Index Name: "language"
Vector Dimension: 2
Similarity Metric: Cosine
Cloud Provider: AWS
Region: us-east-1

⚠️ Limitations

Limited to 2-dimensional vectors
Requires Pinecone serverless deployment
Age matching uses ±1 year range
Returns maximum of 3 similar users

🔒 Security

Store your Pinecone API key securely
Don't commit API keys to version control
Use environment variables for sensitive data

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
