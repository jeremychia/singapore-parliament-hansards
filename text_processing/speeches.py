import pandas as pd
from datetime import date
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
# Evaluation Metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

def get_dates(script_dir, start_date, end_date):
    """
    Retrieve a list of sitting dates within a specified date range.

    Parameters:
    - script_dir (str): The directory path of the script calling this function.
    - start_date (str): The start date of the desired date range (format: 'YYYY-MM-DD').
    - end_date (str): The end date of the desired date range (format: 'YYYY-MM-DD').

    Returns:
    - list of str: A list of sitting dates within the specified date range.

    Example:
    >>> script_directory = os.path.dirname(os.path.abspath(__file__))
    >>> start = '2023-01-01'
    >>> end = '2023-12-31'
    >>> sitting_dates = get_dates(script_directory, start, end)
    >>> print(sitting_dates)
    ['2023-01-02', '2023-02-15', '2023-05-08', ...]
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory, _ = os.path.split(script_dir)

    dates = pd.read_csv(os.path.join(root_directory, 'seeds', 'dates.csv'))
    dates['Sitting_Date'] = pd.to_datetime(dates['Sitting_Date'],
                                           format='%Y-%m-%d')

    date_range_boolean = (dates['Sitting_Date'] >= start_date)\
                         & (dates['Sitting_Date'] <= end_date)

    return list(dates[date_range_boolean]['Sitting_Date'].astype(str))

def get_source_file_path(csv_subfolder):
    """
    Retrieve the absolute file path for a CSV file located in a specified subfolder.

    Parameters:
    - csv_subfolder (str): The name of the subfolder containing the target CSV file.

    Returns:
    - str: The absolute file path for the CSV file.

    Example:
    >>> subfolder_name = 'data_files'
    >>> file_path = get_source_file_path(subfolder_name)

    Note:
    This function assumes that it is called from a Python script located one subfolder layer
    from the root directory.
    """

def get_speech_file_names(source_file_path, dates_to_process):
    """
    Generate a list of speech file names based on the provided source file path and dates.

    Parameters:
    - source_file_path (str): The directory path where the speech files are located.
    - dates_to_process (list of str): A list of dates for which speech file names will be generated.

    Returns:
    - list of str: A list containing the generated speech file names.

    Example:
    >>> source_path = '/path/to/speech_files'
    >>> dates_list = ['2023-01-01', '2023-02-15', '2023-05-08']
    >>> file_names = get_speech_file_names(source_path, dates_list)
    >>> print(file_names)
    ['/path/to/speech_files/2023-01-01-speeches.csv', '/path/to/speech_files/2023-02-15-speeches.csv', '/path/to/speech_files/2023-05-08-speeches.csv']
    """


    speech_file_names  = [os.path.join(source_file_path, f"{date}-speeches.csv")\
            for date in dates_to_process]

    return speech_file_names 

def clean_text(text, additional_stopwords=[]):

    """
    Clean and preprocess the input text by performing the following steps:

    1. Cast text to string.
    2. Convert text to lowercase.
    3. Remove special characters and punctuation.
    4. Remove numbers.
    5. Tokenize the text into words.
    6. Remove standard and additional stop words.
    7. Remove short words (length <= 2).
    8. Lemmatize the words.
    9. Join the cleaned tokens back into text.

    Parameters:
    - text (str): The input text to be cleaned.
    - additional_stopwords (list of str): Additional stop words to be excluded.

    Returns:
    - str: The cleaned and preprocessed text.

    Example:
    >>> input_text = "This is an example text with numbers 123 and special characters! @#"
    >>> additional_stopwords = ['example', 'special']
    >>> cleaned_text = clean_text(input_text, additional_stopwords)
    >>> print(cleaned_text)
    "example text number special character"
    """

    # Cast to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenization (breaking text into words)
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    stop_words.update(additional_stopwords)
    
    tokens = [word for word in tokens if word not in stop_words]

    # Remove short words
    tokens = [word for word in tokens if len(word) > 2]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join the tokens back into text
    cleaned_text = ' '.join(tokens)

    return cleaned_text
    

def get_mp_name(x):
    """
    Extract the Member of Parliament's name from the given input text.

    The function checks if the input text contains the term 'SPEAKER'. If found,
    it attempts to extract the MP's name within parentheses. If not found, it looks for
    common prefixes (e.g., Mr, Mrs, Miss, Mdm, Ms, Dr, Prof) followed by the name.

    Parameters:
    - x (str): The input text containing information about a Member of Parliament.

    Returns:
    - str: The extracted name of the Member of Parliament.

    Example:
    >>> input_text = "SPEAKER (Mr John Doe) called the session to order."
    >>> mp_name = get_mp_name(input_text)
    >>> print(mp_name)
    "John Doe"
    """

    if pd.notna(x) and 'SPEAKER' in x:
        temp = re.search(r'\(([^()]+)\(', x)
        if temp:
            match = re.sub(r'^(?:Mr|Mrs|Miss|Mdm|Ms|Dr|Prof)\s+', '', temp.group(1))
            return match
        else:
            return ''
    elif pd.notna(x):
        match = re.search(r'(?:Mr|Mrs|Miss|Mdm|Ms|Dr|Prof)\s+([\w\s-]+)', x)
        if match:
            return match.group(1)
        else:
            return ''
    else:
        return ''

def prepare_lda_model(cleaned_text):
    """
    Prepare data for Latent Dirichlet Allocation (LDA) model training.

    The function performs the following steps:
    1. Tokenize the cleaned text column.
    2. Create a dictionary representation of the documents.
    3. Convert the tokenized documents into a document-term matrix.

    Parameters:
    - cleaned_text (pd.Series): A Pandas Series containing cleaned and preprocessed text data.

    Returns:
    - tuple: A tuple containing the tokenized text, dictionary, and document-term matrix (corpus).

    Example:
    >>> cleaned_text = pd.Series(["sample text one", "another example text", ...])
    >>> tokenized_text, dictionary, corpus = prepare_lda_model(cleaned_text)
    >>> print(tokenized_text[:2])
    [['sample', 'text', 'one'], ['another', 'example', 'text']]
    >>> print(dictionary)
    Dictionary(7 unique tokens)
    >>> print(corpus[:2])
    [[(0, 1), (1, 1), (2, 1)], [(3, 1), (4, 1), (5, 1)]]
    """
    # Step 1: Tokenize the cleaned text column
    tokenized_text = cleaned_text.apply(lambda x: x.split())

    # Step 2: Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(tokenized_text)

    # Step 3: Convert the tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in tokenized_text]

    return tokenized_text, dictionary, corpus

def intiailise_lda_model(corpus, dictionary, num_topics):
    """
    Initialize and train a Latent Dirichlet Allocation (LDA) model.

    The function applies the LDA model to the provided corpus using the given dictionary.

    Parameters:
    - corpus (list): A list of document-term matrices.
    - dictionary (gensim.corpora.Dictionary): A dictionary representation of the documents.
    - num_topics (int): The number of topics to extract from the corpus.

    Returns:
    - gensim.models.LdaModel: The initialized and trained LDA model.

    Example:
    >>> num_topics = 5
    >>> lda_model = initialise_lda_model(corpus, dictionary, num_topics)
    >>> print(lda_model)
    LdaModel(num_terms=10, num_topics=5, decay=0.5, chunksize=2000)
    """
    # Step 4: Apply the LDA model
    lda_model = models.LdaModel(corpus,
                                num_topics=num_topics,
                                id2word=dictionary,
                                passes=15,
                                random_state=42)

    return lda_model

def evaluate_model(cleaned_text, n_components_range):
    """
    Evaluate the performance of Latent Dirichlet Allocation (LDA) models with different numbers of topics.

    The function uses a range of topic numbers to train LDA models and calculates the log perplexity scores
    for each model, helping to identify an optimal number of topics.

    Parameters:
    - cleaned_text (pd.Series): A Pandas Series containing cleaned and preprocessed text data.
    - n_components_range (list): A list of integers representing the range of topic numbers to evaluate.

    Returns:
    - list: A list of log perplexity scores for each evaluated number of topics.

    Example:
    >>> cleaned_text = pd.Series(["sample text one", "another example text", ...])
    >>> n_components_range = [3, 5, 7]
    >>> perplexity_scores = evaluate_model(cleaned_text, n_components_range)
    >>> print(perplexity_scores)
    [-8.123, -7.512, -7.234]
    """
    documents = cleaned_text.tolist()
    
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(documents)

    tokenized_text, dictionary, corpus = prepare_lda_model(cleaned_text)

    perplexity_scores = []

    for num_topics in n_components_range:

        lda_model = intiailise_lda_model(corpus, dictionary, num_topics)

        # Calculate perplexity score
        perplexity_score = lda_model.log_perplexity(corpus)
        print(f"{num_topics}: log perplexity = {perplexity_score}")
        perplexity_scores.append(perplexity_score)

    return perplexity_scores

def plot_grid_search(n_components_range, perplexity_scores):
    """
    Visualize the results of a grid search for determining the optimal number of topics in LDA.

    The function plots the log perplexity scores against different numbers of topics to aid in
    selecting the optimal number of topics for Latent Dirichlet Allocation (LDA).

    Parameters:
    - n_components_range (list): A list of integers representing the range of topic numbers.
    - perplexity_scores (list): A list of log perplexity scores corresponding to each number of topics.

    Returns:
    - int: A placeholder return value (0).

    Example:
    >>> n_components_range = [3, 5, 7]
    >>> perplexity_scores = [-8.123, -7.512, -7.234]
    >>> plot_grid_search(n_components_range, perplexity_scores)
    (Plots the graph)
    """
    plt.plot(n_components_range, perplexity_scores, marker='o')
    plt.xlabel('Number of Topics')
    plt.ylabel('Log Perplexity')
    plt.title('Optimal Number of Topics for LDA')
    plt.show()

    return 0

def get_optimal_topics(n_components_range, perplexity_scores):
    """
    Determine the optimal number of topics based on log perplexity scores.

    The function identifies the number of topics that results in the maximum log perplexity score,
    considering that log perplexity scores are negative, and values closer to zero (less negative) are better.

    Parameters:
    - n_components_range (list): A list of integers representing the range of topic numbers.
    - perplexity_scores (list): A list of log perplexity scores corresponding to each number of topics.

    Returns:
    - int: The optimal number of topics based on the maximum log perplexity score.

    Example:
    >>> n_components_range = [3, 5, 7]
    >>> perplexity_scores = [-8.123, -7.512, -7.234]
    >>> optimal_topics = get_optimal_topics(n_components_range, perplexity_scores)
    >>> print(optimal_topics)
    7
    """
    optimal_index = perplexity_scores.index(max(perplexity_scores))
    optimal_num_topics = n_components_range[optimal_index]

    return optimal_num_topics

### Variables

csv_subfolder = 'code_output'
script_dir = os.path.dirname(os.path.abspath(__file__))
start_date = '2023-01-01'
end_date = '2023-12-01'
dates_to_process = get_dates(script_dir, start_date, end_date)
additional_stopwords = ['singapore', 'speaker', 'minister', 'asked', 'question',\
                        'please', 'whether', 'year', 'may']
n_components_range = [i+1 for i in range(15)]

### Main run here

speech_files_dir, root_directory = get_source_file_path(csv_subfolder)
speech_files = get_speech_file_names(speech_files_dir, dates_to_process)

dfs = [pd.read_csv(file) for file in speech_files]
df = pd.concat(dfs, ignore_index=True)

## Processing

df['mp_name'] = df['Speaker'].apply(get_mp_name)
df['cleaned_text'] = df['Text']\
                     .apply(lambda x: clean_text(x, additional_stopwords))

## Topics

perplexity_scores = evaluate_model(df['cleaned_text'], n_components_range)
plot_grid_search(n_components_range, perplexity_scores)
optimal_num_topics = get_optimal_topics(n_components_range, perplexity_scores)

tokenized_text, dictionary, corpus = prepare_lda_model(df['cleaned_text'])
lda_model = intiailise_lda_model(corpus, dictionary, optimal_num_topics)
df['topics'] = df['cleaned_text'].apply(lambda x: lda_model[dictionary.doc2bow(x.split())])

for index, row in df.iterrows():
    print(f"Document {index + 1}: {row['cleaned_text']}")
    print("Topics:")
    for topic, score in row['topics']:
        print(f"  Topic {topic + 1}: {lda_model.print_topic(topic)} (Score: {score:.4f})")
    print("\n")

    if index > 5:
        break




