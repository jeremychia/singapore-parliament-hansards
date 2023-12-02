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


def get_source_file_path(csv_subfolder):
    '''
    This assumes that the Python Script
    is in one subfolder layer from the root.
    '''

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_directory, _ = os.path.split(script_dir)

    return os.path.join(root_directory, csv_subfolder)

def get_speech_file_names(source_file_path, dates_to_process):

    temp = [source_file_path+'\\'+date+'-speeches.csv'\
            for date in dates_to_process]

    return temp

def clean_text(text, additional_stopwords):
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

def intiailise_lda_model(cleaned_text, num_topics):

    # Step 1: Tokenize the cleaned text column
    tokenized_text = cleaned_text.apply(lambda x: x.split())

    # Step 2: Create a dictionary representation of the documents
    dictionary = corpora.Dictionary(tokenized_text)

    # Step 3: Convert the tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in tokenized_text]

    # Step 4: Apply the LDA model
    lda_model = models.LdaModel(corpus,\
                                num_topics=num_topics,\
                                id2word=dictionary,\
                                passes=15,\
                                random_state=42)

    return lda_model, dictionary, corpus

def evaluate_model(cleaned_text, n_components_range):

    documents = cleaned_text.tolist()
    
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(documents)

    perplexity_scores = []

    for num_topics in n_components_range:

        lda_model, _, corpus = intiailise_lda_model(cleaned_text, num_topics)

        # Calculate perplexity score
        perplexity_score = lda_model.log_perplexity(corpus)
        print(f"{num_topics}: log perplexity = {perplexity_score}")
        perplexity_scores.append(perplexity_score)

    return perplexity_scores

def plot_grid_search(n_components_range, perplexity_scores):

    plt.plot(n_components_range, perplexity_scores, marker='o')
    plt.xlabel('Number of Topics')
    plt.ylabel('Log Perplexity')
    plt.title('Optimal Number of Topics for LDA')
    plt.show()

    return 0

### Variables

csv_subfolder = 'code_output'
script_dir = os.path.dirname(os.path.abspath(__file__))
dates_to_process = ['2023-11-22']
additional_stopwords = ['singapore', 'speaker', 'minister', 'asked', 'question',\
                        'please', 'whether', 'year', 'may']
n_components_range = [i+1 for i in range(15)]

### Main run here

speech_files_dir = get_source_file_path(csv_subfolder)
speech_files = get_speech_file_names(speech_files_dir, dates_to_process)

df = pd.read_csv(speech_files[0])

## Processing

df['mp_name'] = df['Speaker'].apply(get_mp_name)
df['cleaned_text'] = df['Text']\
                     .apply(lambda x: clean_text(x, additional_stopwords))

## Topics

perplexity_scores = evaluate_model(df['cleaned_text'], n_components_range)
plot_grid_search(n_components_range, perplexity_scores)

lda_model, dictionary, corpus = intiailise_lda_model(df['cleaned_text'], 5)
df['topics'] = df['cleaned_text'].apply(lambda x: lda_model[dictionary.doc2bow(x.split())])

for index, row in df.iterrows():
    print(f"Document {index + 1}: {row['cleaned_text']}")
    print("Topics:")
    for topic, score in row['topics']:
        print(f"  Topic {topic + 1}: {lda_model.print_topic(topic)} (Score: {score:.4f})")
    print("\n")

    if index > 5:
        break




