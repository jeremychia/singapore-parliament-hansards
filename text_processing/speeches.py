import pandas as pd
from datetime import date
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models

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

def clean_text(text):
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



### Variables

csv_subfolder = 'code_output'
script_dir = os.path.dirname(os.path.abspath(__file__))
dates_to_process = ['2023-11-22']

### Main run here

speech_files_dir = get_source_file_path(csv_subfolder)
speech_files = get_speech_file_names(speech_files_dir, dates_to_process)

df = pd.read_csv(speech_files[0])

## Processing

df['MP_Name'] = df['Speaker'].apply(get_mp_name)
df['Cleaned_Text'] = df['Text'].apply(clean_text)

## Topics

# Step 1: Tokenize the cleaned text column
tokenized_text = df['Cleaned_Text'].apply(lambda x: x.split())

# Step 2: Create a dictionary representation of the documents
dictionary = corpora.Dictionary(tokenized_text)

# Step 3: Convert the tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in tokenized_text]

# Step 4: Apply the LDA model
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15, random_state=42)

# Step 5: Extract topics for each document
df['Topics'] = df['Cleaned_Text'].apply(lambda x: lda_model[dictionary.doc2bow(x.split())])

for index, row in df.iterrows():
    print(f"Document {index + 1}: {row['Cleaned_Text']}")
    print("Topics:")
    for topic, score in row['Topics']:
        print(f"  Topic {topic + 1}: {lda_model.print_topic(topic)} (Score: {score:.4f})")
    print("\n")

    if index > 5:
        break




