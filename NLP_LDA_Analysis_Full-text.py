"""
Created on Wed Feb 04 2026
@author: Sander Marko Kruse

PDF Full-Text NLP - LDA Pipeline
----------------------------------
A pipeline designed to transform raw PDF data into structured 
thematic insights using Natural Language Processing (NLP) and Latent 
Dirichlet Allocation (LDA).

It automates the following workflow:
1. Data Extraction: Scans a predefined directorie and extracts raw text from PDF files, so computer can process.
2. Preprocessing: Cleans and tokenizes text for downstream analysis.
3. Exploratory Analysis: Generates Word Frequencies, TF-IDF scores, and Bigrams.
4. Evaluation: Is done through a manual grid search to evalute hyperparameters (alpha, 
   beta, topics, top words per topic) and stores model parameters and results and metrics (Perplexity, Coherence, Vocab).
5. Topic Modeling: Runs Latent Dirichlet Allocation (LDA) via Scikit-Learn based on the optimal model found.
6. Visualization: Final model representation is done via UMAP, top words per topic


Last edited: 10.03.2026
"""

#%% Importing useful packages
from pathlib import Path                                                     # Reading in path of files
import fitz                                                                  # PDF reader
import pandas as pd                                                          # For creating dataframes
import re                                                                    # Regular expression, used for data cleaning
from tqdm import tqdm                                                        # Used for progress bar when reading in files
#from pypdf import PdfReader
import nltk                                                                  # For extracting stopwords from a database
from nltk.corpus import stopwords                                            # Extraction of stopword
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # Used for frequencies of words
import matplotlib.pyplot as plt                                              # Used for visualisation
from wordcloud import WordCloud                                              # Used for visualisation
import seaborn as sns                                                        # Seaborn used for barplots
from sklearn.decomposition import LatentDirichletAllocation                  # LDA for topic modeling
from nltk.stem import WordNetLemmatizer

#%% Downloading stopwords from NLTK
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Download WordNet data and POS tagger
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = WordNetLemmatizer()

#%% Reading in the files from folder. Adjust accordingly to the folders you want to read in
pdf_folders = [
    Path(r"Samling av dokumenter")
]

# Creating a function to extract text from PDF by using pypdf
def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            page_text = []
            for page in doc:
                page_text.append(page.get_text("text"))
                
            return "\n".join(page_text).strip()
    except Exception as e1:
        print(f"Error reading {pdf_path.name}: {e1}")
        return ""

# Collecting all PDFs
pdf_files = []
for folder in pdf_folders:
    try:
        for file in folder.rglob("*pdf"):
            pdf_files.append(file)
    except Exception as e2:
        print(f"Error reading in the files. \nExxeption error {e2}")

print(f"Found {len(pdf_files)} PDF files across folders")


# Adding a progress bar to see how many articles are read in
pbar = tqdm(
    pdf_files, 
    desc= "[Processing Articles]", 
    #unit= "article",
    colour="green",
    ncols=130  # Setting a fixed width
)

# Create DataFrames. Adding in three columns, filename, raw_text and path
data = []
test = []
for pdf_path in pbar:
    # Short name of the PDFs (Used for visualisation)
    short_name = (pdf_path.name[:27] + '..') if len(pdf_path.name) > 30 else pdf_path.name
    pbar.set_description(f"Current: {short_name}")
    
    raw_text = extract_text_from_pdf(pdf_path)
    
    if raw_text:
        data.append({
            'filename': pdf_path.name,
            'raw_text': raw_text,
            'path': str(pdf_path)
        })

df_pdfs = pd.DataFrame(data)
print(f"\nSuccessfully extracted text from {len(df_pdfs)} PDFs")
#%%
df_pdfs['filename_norm'] = df_pdfs['filename'].astype(str).str.strip().str.lower()
print("Checking for duplicates within the filenames.")
df_pdfs = df_pdfs.drop_duplicates(subset=['filename_norm'], keep='first')
print('\n............Checking..........\n')
print(f"Removed {len(data) - len(df_pdfs)} duplicated entries. \nTotal PDFs: {len(df_pdfs)}")

#%% Text cleaning function
def clean_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    
    # 1. Isolating the content. Keeping text from abstract to the first finding of ref_patterns
    # Removing everythin until abstract is found
    abstract_patterns = [
    r'abstract\s*:?',
    r'ABSTRACT\s*:?',
    r'a\s*b\s*s\s*t\s*r\s*a\s*c\s*t\s*:?',
    r'A B S T R A C T:?',
    ]
    
    abstract_start = -1
    text_lower = text.lower()
    
    for pattern in abstract_patterns:
        match = re.search(pattern, text_lower)
        if match:
            pos = match.start()
            # Safety: prefer early matches (first 5000 chars)
            if pos < 5000:
                abstract_start = pos
                break
    
    if abstract_start == -1:
        abstract_start = 0  # fallback
    
    # Take text from abstract onward
    text = text[abstract_start:]
    
    
    # This step cuts the text at the FIRST occurrence of a noisy heading found in the ref_patterns list below.
    # Meaning, if declaration of competing interest is the first occurence of the PDF. All text after is removed.
    # Since Science Direct and research articles follow a similar/standard tempelate, i have made the occurrences line up to how the papers are mostly produced.
    ref_patterns = [
        r"\n\s*CRediT authorship contribution statement\s*\n",
        r"\n\s*Declaration of competing interest\s*\n",
        r"\n\s*Acknowledgment\s*\n",         
        r"\n\s*Acknowledgements\s*\n",        
        r"\n\s*BIBLIOGRAPHY\s*\n",
        r"\n\s*Literature Cited\s*\n",
        r"\n\s*Reference List\s*\n",
        r"\n\s*References",               
        r"\n\s*Cited References\s*\n"
    ]
    
    
    cut_position = len(text)  # default: keep everything

    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            # Only use the **earliest** match that is late in the document
            if match.start() > len(text) * 0.7:  # only if after ~60 % of text
                cut_position = min(cut_position, match.start())
    
    if cut_position < len(text):
        text = text[:cut_position]
    
    
    # 2. Section for Standard cleaning
    
    # Lowercasing all the words
    text = text.lower()
    
    #
    ieee_watermark = r"\d+\s+Authorized licensed use limited to:.*?Restrictions apply\."
    text = re.sub(ieee_watermark, "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove citations anywhere in the text. Such as [1], [1, 2]
    text = re.sub(r'\[\d+[,\s\-\d]*\]', '', text)
    
    # Remove boilerplate and copyright
    text = re.sub(r'©.*?(elsevier|ieee|all rights reserved).*?$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'published by.*', '', text, flags=re.IGNORECASE)
    
    # Remove DOI, URLs and common metadata
    text = re.sub(r'(doi|https?://|ieee|org|trans|fig\.?|figure\s*\d+)[^\s]*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(doi|url|reference|references|bib|ref)\b.*', ' ', text, flags=re.IGNORECASE)
    
    # Remove lines that look like bibliography entries (e.g. start with number or [1])
    text = re.sub(r'^\s*\[\d+\].*$', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s.*$', ' ', text, flags=re.MULTILINE)
    
    # Remove caption, such as in figures and table
    text = re.sub(r'(shown\s+in|see|as\s+shown\s+in)\s*(fig\.?|figure|table|tab)\s*\d+', ' ', text, flags=re.IGNORECASE)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s-]', ' ', text)
    
    # Collapsing multiple space and stipping the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    
    # 3. Tokenization and stopwords (Remove stopwords and short words)
    tokenized_words = text.split()
    
    # Lemmatizing the words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokenized_words]
    
    cleaned_words = [token for token in lemmatized_words if token not in stop_words and len(token) > 2]
    
    return ' '.join(cleaned_words)

# Apply cleaned text to the dataframe
df_pdfs['clean_text'] = df_pdfs['raw_text'].apply(clean_text)
print("Text cleaning completed.")

#%%
# Find the paper by title (partial match is fine)
paper_title = "Fault_Detection_of_Hydroelectric_Generators_using_Isolation_Forest"

matching_rows = df_pdfs[df_pdfs['filename'].str.contains(paper_title, case=False, na=False)]

if len(matching_rows) == 0:
    print("Paper not found. Try a shorter part of the title.")
else:
    idx = matching_rows.index[0]
    print(f"Found paper at index: {idx}")
    print(f"Title: {df_pdfs.loc[idx, 'filename']}")
    print(f"Retained percentage: {df_pdfs.loc[idx, 'retained_pct']}%")
    
    raw_text = str(df_pdfs.loc[idx, 'raw_text'])
    clean_text = str(df_pdfs.loc[idx, 'clean_text'])
    
    print("\n" + "="*80)
    print("RAW TEXT - FIRST 1000 CHARACTERS:")
    print("="*80)
    print(raw_text[:1000])
    
    print("\n" + "="*80)
    print("CLEANED TEXT - FIRST 1000 CHARACTERS:")
    print("="*80)
    print(clean_text[:1000])
    
    print("\n" + "="*80)
    print("Siste ordene")
    print("RAW TEXT - FIRST 1000 CHARACTERS:")
    print("="*80)
    print(raw_text[-13000:])
    
    print("\n" + "="*80)
    print("CLEANED TEXT - FIRST 1000 CHARACTERS:")
    print("="*80)
    print(clean_text[-13000:])
    

#%% Checking how much is deleted frorm the corpora
df_pdfs['raw_length'] = df_pdfs['raw_text'].astype(str).apply(len)
df_pdfs['clean_length'] = df_pdfs['clean_text'].astype(str).apply(len)

df_pdfs['retained_pct'] = (df_pdfs['clean_length'] / df_pdfs['raw_length'].replace(0, 1)) * 100
df_pdfs['retained_pct'] = df_pdfs['retained_pct'].round(1)

print("=== Quick Length Comparison Summary ===")
print(df_pdfs[['raw_length', 'clean_length', 'retained_pct']].describe().round(1))

print(f"\nAverage kept: {df_pdfs['retained_pct'].mean():.1f}%")
print(f"Median kept:  {df_pdfs['retained_pct'].median():.1f}%")
print(f"Papers <30% kept: {(df_pdfs['retained_pct'] < 30).sum()}")
print(f"Papers <10% kept: {(df_pdfs['retained_pct'] < 10).sum()}")

worst = df_pdfs[df_pdfs['retained_pct'] < 50].sort_values('retained_pct')
for idx, row in worst.iterrows():
    print(f"\nPaper: {row['filename'][:80]}...")
    print(f"Retained: {row['retained_pct']}%")
    print("Raw start:", row['raw_text'][:300])
    print("Clean start:", row['clean_text'][:300])
    
#%% Printing out the last 1000 characters of the PDF to see if references and acknowledgment is gone.

for i in range(min(10, len(df_pdfs))):
    filename = df_pdfs['filename'].iloc[i]
    raw = df_pdfs['raw_text'].iloc[i]
    clean = df_pdfs['clean_text'].iloc[i]
    
    # Show first 5000 characters of raw and cleaned version.
    raw_preview = raw[:1000] + "..." if len(raw) > 1000 else raw
    clean_preview = clean[:1000] + "..." if len(clean) > 1000 else clean
    
    print(f"\nPDF {i+1}: {filename}")
    print("-"*80)
    print("RAW TEXT (first part):")
    print(raw_preview)
    print("\nCLEANED TEXT (first part):")
    print(clean_preview)
    print("-"*80)

#%% Checking the corpora to see if more cleaning is needed
last_10_df = df_pdfs.tail(10)

# Iterate through the subset
for i, (index, row) in enumerate(last_10_df.iterrows()):
    filename = row['filename']
    raw = row['raw_text']
    clean = row['clean_text']
    
    # Slicing the last 10,000 characters for the preview
    raw_preview = raw[-10000:] + "..." if len(raw) > 10000 else raw
    clean_preview = clean[-10000:] + "..." if len(clean) > 10000 else clean
    
    print(f"\nDocument {i+1} (Original Index {index}): {filename}")
    print("-" * 80)
    print("RAW TEXT PREVIEW (Last 10k chars):")
    print(raw_preview)
    print("\nCLEANED TEXT PREVIEW (Last 10k chars):")
    print(clean_preview)
    print("-" * 80)
    
#%% Visualisation of WordCloud, with no filters applied on CountVectorizer

vectorizer_count = CountVectorizer()
X_count = vectorizer_count.fit_transform(df_pdfs['clean_text'])

word_freq = pd.DataFrame(
    X_count.sum(axis=0).T,
    index=vectorizer_count.get_feature_names_out(),
    columns=['count']
).sort_values('count', ascending=False)

print("\n=== Top 30 most frequent words across all abstracts ===")
print(word_freq.head(30))

# Visualization of WordCloud
wordcloud = WordCloud(
    width=1200, height=600,
    background_color='white',
    max_words=100,
    colormap='viridis'
).generate_from_frequencies(word_freq['count'].to_dict())

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Abstracts (frequency)")
plt.show()

#%% Visualisation of unigram, bigrams and comparison on a shared scale between TF-IDF and CountVectorizer
def visualize_nlp_results(df, text_col, mode='count', n_range=(1, 1), top_n=20, title="Top Terms"):
    """
    Modes: 
    'count'      : Standard bar plot of raw frequencies.
    'tfidf'      : Standard bar plot of TF-IDF importance.
    'compare'    : Dual-bar comparison (Normalized 0-1).
    """
    # Setup Vectorizers
    cv = CountVectorizer(ngram_range=n_range, max_df=0.8, min_df=2, stop_words=None)
    tfidf_vec = TfidfVectorizer(ngram_range=n_range, max_df=0.80, min_df=2, stop_words=None)
    
    # Process Data
    X_c = cv.fit_transform(df[text_col])
    X_t = tfidf_vec.fit_transform(df[text_col])
    
    # Create DataFrames
    c_df = pd.DataFrame(X_c.sum(axis=0).T, index=cv.get_feature_names_out(), columns=['count'])
    t_df = pd.DataFrame(X_t.mean(axis=0).T, index=tfidf_vec.get_feature_names_out(), columns=['tfidf'])

    plt.figure(figsize=(10, 7))

    # Deciding what to plot
    if mode == 'count':
        data = c_df.sort_values('count', ascending=False).head(top_n).reset_index()
        sns.barplot(data=data, x='count', y='index', palette='mako')
        plt.xlabel("Raw Frequency Count")

    elif mode == 'tfidf':
        data = t_df.sort_values('tfidf', ascending=False).head(top_n).reset_index()
        sns.barplot(data=data, x='tfidf', y='index', palette='viridis')
        plt.xlabel("Mean TF-IDF Score")

    elif mode == 'compare':
        # Anchor on TF-IDF, then lookup counts
        combined = t_df.sort_values('tfidf', ascending=False).head(top_n).copy()
        combined['raw_count'] = c_df.loc[combined.index, 'count']
        
        # Normalization
        combined['TF-IDF (Norm)'] = combined['tfidf'] / combined['tfidf'].max()
        combined['Occurence (Norm)'] = combined['raw_count'] / combined['raw_count'].max()
        
        plot_df = combined[['TF-IDF (Norm)', 'Occurence (Norm)']].reset_index().melt(id_vars='index')
        sns.barplot(data=plot_df, x='value', y='index', hue='variable', palette='viridis')
        plt.xlabel("Relative weight (Normalized 0-1)")

    #plt.title(f"{title} ({mode.upper()})", fontsize=14)
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
    
# Unigram Counts
visualize_nlp_results(df_pdfs, 'clean_text', mode='count', n_range=(1,1), title="Top Unigrams")

# Bigram Counts
visualize_nlp_results(df_pdfs, 'clean_text', mode='count', n_range=(2,2), title="Top Bigrams")

# Unigram TF-IDF
visualize_nlp_results(df_pdfs, 'clean_text', mode='tfidf', n_range=(1,1), title="TF-IDF Unigrams")

# Bigram TF-IDF
visualize_nlp_results(df_pdfs, 'clean_text', mode='tfidf', n_range=(2,2), title="TF-IDF Bigrams")

# Bigram Comparison
visualize_nlp_results(df_pdfs, 'clean_text', mode='compare', n_range=(2,2), top_n=10, title="Bigram Context")





#%% Tuning of hyperparameteres of LDA, across different combinations of alpha and betas, with different top words per topic

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# Defenition to merging the multi words
def merge_phrases(text, phrases_dict):
    # Replace longer phrases first
    for phrase, token in sorted(phrases_dict.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r'\b' + re.escape(phrase) + r'\b'
        text = re.sub(pattern, token, text)
    return text

# Applying the second clean from the domain-specific words
def second_clean(text):
    if not text.strip():
        return ""
    words = text.split()
    cleaned = [w for w in words if w not in domain_specific_stops]
    return ' '.join(cleaned)

# Defenition to comunting coherence score
def compute_coherence(lda_model, feature_names, top_n=10):
    topics = []
    for topic_weights in lda_model.components_:
        top_idx = topic_weights.argsort()[-top_n:][::-1]
        top_words = [feature_names[i] for i in top_idx]
        topics.append(top_words)
        
    cm = CoherenceModel(
        topics = topics,
        texts = texts,
        dictionary = dictionary,
        coherence = 'u_mass', # Change this to c_v or use u_mass
        processes=1
        )
    return cm.get_coherence(), topics

# Using the knowledge obtained from the bigrams to create a multi word phrase to one token
multi_word_phrases = {
    # Picked from top 20 bigrams
    'neural network': 'neural_network',
    'renewable energy': 'renewable_energy',
    'real time': 'real_time',
    'hydropower plant': 'hydropower_plant',
    'fault diagnosis': 'fault_diagnosis',
    'machine learning': 'machine_learning',
    'short term': 'short_term',
    'power system': 'power_system',
    'fault detection': 'fault_detection',
    'time series': 'time_series',
    'deep learning': 'deep_learning',
    'power plant': 'power_plant',
    'hydro turbine': 'hydro_turbine',
    'predictive maintenance': 'predictive_maintenance',
    'cnn lstm': 'cnn_lstm',
    'condition monitoring': 'condition_monitoring',
    'feature extraction': 'feature_extraction',
    'power generation': 'power_generation',
    
    # Other
    'acoustic signals': 'acoustic_signals',
    'anomaly detection': 'anomaly_detection',
    'hydropower unit': 'hydropower_unit',
    'convolutional neural': 'cnn',
    'long short term': 'lstm',
    'support vector machine': 'svm',
    'random forest': 'random_forest',
    'wavelet transform': 'wavelet_transform'
    
}

df_pdfs['clean_text'] = df_pdfs['clean_text'].apply(lambda x: merge_phrases(x, multi_word_phrases))

# Creating domain specific stopwords, that is applied to the text.
domain_specific_stops = {  
    'paper', 'study', 'proposed', 'proposes', 'presents',
    'using', 'results', 'result', 'show', 'shown',
    'therefore', 'however', 'thus', 'respectively',
    'framework', 'key', 'research', 'researches',
    'analysis', 'show'
}

df_pdfs['clean_text'] = df_pdfs['clean_text'].apply(second_clean)

# Using Gensim dictionary for coherence
texts = df_pdfs['clean_text'].str.split().tolist()
dictionary = corpora.Dictionary(texts)

# Manually setting Parameter ranges
topics_range = range(1, 8, 1)
alpha_range = [0.01, 0.1, 0.5, 0.7, 1, None]
beta_range = [0.01, 0.1, 0.3, 0.5, 1, None]
min_df_range  = [2]
max_df_range  = [0.80]
number_of_words = [4, 5, 7, 10]

# Creating dictonary to store values
model_results = {
    'Topics': [],                    # Total number of themes the model is to find
    'Alpha': [],                     # Document - topic density
    'Beta': [],                      # Topic-word density
    'min_df': [],                    # Minimum document frequency for a word to be included
    'max_df': [],                    # Maximum document frequency
    'top_n_words': [],               # Number of words used to represent each topic
    'Coherence': [],                 # Logic score
    'Perplexity': [],                # Statistical score
    'Vocab_Size': [],                # Total uniqe words used after filtering
    'Top_words_per_topic': []        # Actual keywords that defines each discovered theme
}

# Calculating the total combinations to be evaluted across the manual grid search
total_combinations = (
    len(topics_range) * len(alpha_range) * len(beta_range) *
    len(min_df_range) * len(max_df_range) * len(number_of_words)
)

print(f"Evaluating {total_combinations} combinations...")

with tqdm(total = total_combinations, desc = "LDA tuning") as pbar:
    for n_topics in topics_range:
        for alpha in alpha_range:
            for beta in beta_range:
                for min_df in min_df_range:
                    for max_df in max_df_range:
                        for n_words in number_of_words:
                        
                            vectorizer = CountVectorizer(
                                min_df = min_df,
                                max_df = max_df,
                                stop_words = None,
                                )
                            
                            X = vectorizer.fit_transform(df_pdfs['clean_text'])
                            
                            lda = LatentDirichletAllocation(
                                
                                n_components=n_topics,
                                doc_topic_prior=alpha,
                                topic_word_prior=beta,
                                max_iter=100,
                                learning_method='batch',
                                random_state=42,
                                n_jobs=-1
                                )
                            
                            lda.fit(X)
                            
                            # ── Compute coherence ────────────────────────────────
                            
                            feature_names = vectorizer.get_feature_names_out()
                            
                            
                            coherence, current_topics = compute_coherence(
                                lda, 
                                vectorizer.get_feature_names_out(),
                                top_n=n_words)
                            
                            perplexity = lda.perplexity(X)
                            
                            model_results['Topics'].append(n_topics)
                            model_results['Alpha'].append(alpha if alpha is not None else 'symmetric')
                            model_results['Beta'].append(beta if beta is not None else 'symmetric')
                            model_results['min_df'].append(min_df)
                            model_results['max_df'].append(max_df)
                            model_results['top_n_words'].append(n_words)          
                            model_results['Perplexity'].append(perplexity)
                            model_results['Vocab_Size'].append(X.shape[1])
                            model_results['Coherence'].append(coherence)
                            
                            # store the words (e.g. as pipe-separated string)
                            topic_strings = [" ".join(words) for words in current_topics]
                            all_topics_str = " | ".join(topic_strings)
                            model_results['Top_words_per_topic'].append(all_topics_str)
                            
                            pbar.update(1)
# Saving the results
df_results = pd.DataFrame(model_results)

#%% Saving the results to a CSV file

df_results.to_csv('lda_model_fulltext_results_UMass_V7_medstopwords_batch_max_df=0.80V3.csv', sep=';', index=False)

# Printing out the top 10 models by highest coherence

print("\nTop 10 models by highest coherence:")
print(df_results.sort_values('Coherence', ascending=False).head(10))

#%% Visualizing coherence scores across alphas and betas, creating a faceit grid showing all combinations

# Filter for all desired word counts at once
df_facet = df_results[df_results['top_n_words'].isin(number_of_words)]

# Add hue to FacetGrid to separate the 4 lines by word count
g = sns.FacetGrid(df_facet, col="Beta", row="Alpha", hue="top_n_words", margin_titles=True)

# Map the lineplot 
g.map(sns.lineplot, "Topics", "Coherence", marker="o")
g.map(plt.grid, color='lightgrey', linestyle='--')

# Adding a legend for number of words
g.add_legend(title="Word Combinations", adjust_subtitles=True)
sns.move_legend(g, loc="upper left", frameon=True)
                
# Setting limits on axis and styling the plot
g.set(ylim=(-1, 0.01), yticks=np.arange(-1, 0.01, 0.2))
g.set_axis_labels("Topics", "UM")
g.fig.suptitle("Coherence Peaks across Alpha/Beta & Word Counts", fontsize=13, y=0.97)
g.fig.tight_layout(rect=[0, 0.02, 1, 0.92])

plt.show()

#%% Plotting the top words from the final LDA model
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 4, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 15})
        ax.tick_params(axis="both", which="major", labelsize=10)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=25)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

vectorizer_lda = CountVectorizer(max_df=0.8, min_df=2, stop_words=None)  # ignore common words, max_df
X_lda = vectorizer_lda.fit_transform(df_pdfs['clean_text'])

n_topics = 5  # Change here for the number of topics

lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=100,
    learning_method='batch',
    random_state=42,
    n_jobs=-1,
    doc_topic_prior=1,
    topic_word_prior=0.3
)

lda.fit(X_lda)

def display_topics(model, feature_names, n_top_words=4):
    for i, topic in enumerate(model.components_):
        print(f"Topic {i+1}: {' '.join([feature_names[j] for j in topic.argsort()[-n_top_words:][::-1]])}")

print("\n=== LDA Topics (top 10 words each) ===\n")
display_topics(lda, vectorizer_lda.get_feature_names_out())


tf_feature_names = vectorizer_lda.get_feature_names_out()
plot_top_words(lda, tf_feature_names, 5, title="Topics in LDA model")


#%% Creating UMAPS for different K neighbours, 5,10,15,20
import numpy as np
import matplotlib.pyplot as plt
import umap

# Find the topic probability ditributions
doc_topics = lda.transform(X_lda)


# Test noen få n_neighbors-verdier
n_values = [5, 10, 15, 20]

plt.figure(figsize=(12, 10))

for i, n in enumerate(n_values, 1):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n,
        min_dist=0.1,
        random_state=42
    )
    doc_2d = reducer.fit_transform(doc_topics)
    
    plt.subplot(2, 2, i)
    scatter = plt.scatter(
        doc_2d[:, 0], doc_2d[:, 1],
        c=np.argmax(doc_topics, axis=1),
        cmap='tab10',
        s=50,
        alpha=0.7,
        edgecolor='none'
    )
    plt.title(f'n_neighbors = {n}')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('UMAP Projections – Colored by Dominant LDA Topic', y=1.02)
plt.colorbar(scatter, ax=plt.gcf().axes, label='Dominant Topic')
plt.show()

#%% Picking the final best solution for visualizing in UMAP

doc_topics = lda.transform(X_lda)

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,          # Best choice
    min_dist=0.1,
    random_state=42
)
doc_2d = reducer.fit_transform(doc_topics)




# Dine topic-navn (endre til dine endelige)
topic_labels = [
    "Hydro Temp & Fault Detection",         # Topic 1
    "Hydropower Management & Grid",         # Topic 2
    "Signal Processing & DL Prediction",    # Topic 3
    "Wind/Renewable Forecasting ",          # Topic 4   
    "Turbine Vibration & Bearing Fault"     # Topic 5
]

# Plot
plt.figure(figsize=(8, 6))

dominant = np.argmax(doc_topics, axis=1)
scatter = plt.scatter(
    doc_2d[:, 0], doc_2d[:, 1],
    c=dominant,
    cmap='viridis',
    s=50,
    alpha=0.7,
    edgecolor='none'
)

# Placing of the legend
cbar = plt.colorbar(scatter, ticks=range(5), orientation='vertical')
cbar.ax.set_yticklabels(topic_labels)  


plt.title('UMAP Projection of Full text')
plt.xlabel('UMAP dimension 1')
plt.ylabel('UMAP dimension 2')
plt.grid(True, alpha=0.3)
#plt.tight_layout()
plt.show()