import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import sys

# Check for required dependencies
required_modules = ['pandas', 'nltk', 'matplotlib', 'seaborn', 'pyparsing', 'Pillow', 'packaging', 'regex', 'pytz', 'dateutil']
missing_modules = []
for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        missing_modules.append(module)
if missing_modules:
    sys.exit(f"Error: Missing required modules: {missing_modules}. Install them using 'pip install {' '.join(missing_modules)}'")

# Download NLTK data quietly
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def load_data(file_path='data/raw_analyst_rating.csv'):
    """Load the analyst rating dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file (default: 'data/raw_analyst_rating.csv').
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    
    Raises:
        FileNotFoundError: If the file or directory is not found.
        ValueError: If data loading fails.
    """
    data_dir = os.path.dirname(file_path)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please create it and place 'raw_analyst_rating.csv' there.")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found in '{data_dir}'. Please check the path.")
    
    try:
        df = pd.read_csv(file_path)
        expected_columns = ['headline', 'publisher', 'date', 'stock']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing expected columns {missing_columns}. Available columns: {list(df.columns)}")
        return df
    except Exception as e:
        raise ValueError(f"Error loading data from '{file_path}': {str(e)}")

def descriptive_stats(df):
    """Compute descriptive statistics for headline lengths and article counts.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'headline' and 'publisher' columns.
    
    Returns:
        tuple: Descriptive statistics of headline lengths and publisher article counts.
    """
    if 'headline' not in df.columns:
        raise ValueError(f"Column 'headline' not found in dataset. Available columns: {list(df.columns)}")
    
    # Compute headline length, handling non-string values
    df['headline_length'] = df['headline'].apply(lambda x: len(str(x)))
    stats = df['headline_length'].describe()
    print("Headline Length Statistics:")
    print(stats)
    
    # Count articles per publisher
    publisher_counts = None
    if 'publisher' in df.columns:
        publisher_counts = df['publisher'].value_counts()
        print("\nArticles per Publisher (Top 10):")
        print(publisher_counts.head(10))
    else:
        print("Warning: 'publisher' column not found. Skipping publisher counts.")
    
    return stats, publisher_counts

def time_series_analysis(df):
    """Analyze publication frequency over time.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'date' column.
    
    Returns:
        pd.Series: Daily article counts.
    """
    if 'date' not in df.columns:
        raise ValueError(f"Column 'date' not found in dataset. Available columns: {list(df.columns)}")
    
    try:
        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
        if df['date'].isna().all():
            raise ValueError("All 'date' values are invalid or could not be parsed.")
        df['date_only'] = df['date'].dt.date
    except Exception as e:
        raise ValueError(f"Error converting 'date' column to datetime: {str(e)}")
    
    daily_counts = df.groupby('date_only').size()
    
    plt.figure(figsize=(12, 6))
    daily_counts.plot(color='#1f77b4')
    plt.title('Article Publication Frequency Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('outputs/publication_frequency.png', dpi=300)
    plt.close()
    
    return daily_counts

def text_analysis(df):
    """Perform text analysis on headlines to identify common words.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'headline' column.
    
    Returns:
        list: Top 20 most common words and their counts.
    """
    if 'headline' not in df.columns:
        raise ValueError(f"Column 'headline' not found in dataset. Available columns: {list(df.columns)}")
    
    stop_words = set(stopwords.words('english'))
    all_words = []
    
    for headline in df['headline']:
        if not isinstance(headline, str):
            continue
        words = word_tokenize(str(headline).lower())
        words = [w for w in words if w.isalpha() and w not in stop_words]
        all_words.extend(words)
    
    word_freq = Counter(all_words)
    common_words = word_freq.most_common(20)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[count for word, count in common_words], 
                y=[word for word, count in common_words], 
                hue=[word for word, count in common_words], 
                palette='viridis', 
                legend=False)
    plt.title('Top 20 Common Words in Headlines')
    plt.xlabel('Frequency')
    plt.ylabel('Word')
    plt.savefig('outputs/common_words.png', dpi=300)
    plt.close()
    
    return common_words

def publisher_domain_analysis(df):
    """Analyze unique domains from publisher emails.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'publisher' column.
    
    Returns:
        pd.Series: Counts of articles per publisher domain.
    """
    if 'publisher' not in df.columns:
        print(f"Warning: 'publisher' column not found. Skipping domain analysis. Available columns: {list(df.columns)}")
        return None
    
    df['domain'] = df['publisher'].str.extract(r'@([\w\.-]+)')
    domain_counts = df['domain'].value_counts()
    
    if domain_counts.empty:
        print("No valid domains extracted from publisher column.")
        return domain_counts
    
    plt.figure(figsize=(12, 6))
    domain_counts.head(10).plot(kind='bar', color='#2ca02c')
    plt.title('Top 10 Publisher Domains')
    plt.xlabel('Domain')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('outputs/publisher_domains.png', dpi=300)
    plt.close()
    
    return domain_counts

if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    try:
        df = load_data()
        stats, publisher_counts = descriptive_stats(df)
        daily_counts = time_series_analysis(df)
        common_words = text_analysis(df)
        domain_counts = publisher_domain_analysis(df)
        print("\nEDA Completed. Visualizations saved as PNG files in the 'outputs' directory.")
    except Exception as e:
        print(f"Error during EDA: {str(e)}")