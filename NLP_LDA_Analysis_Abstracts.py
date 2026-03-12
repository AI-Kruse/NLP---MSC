"""
Created on Wed Feb 04 2026
@author: Sander Marko Kruse

Abstract NLP - LDA Pipeline
----------------------------------
A pipeline designed to transform raw .ris file data into structured 
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
#%%
import rispy                                                                    # Reading in .ris files
from pathlib import Path                                                        # Reading in path of files
import pandas as pd                                                             # For creating dataframes
import re                                                                       # Regular expression, used for data cleaning
from nltk.corpus import stopwords                                               # For extracting stopwords from a database
import nltk                                                                     # For extracting stopwords from a database
from nltk.stem import WordNetLemmatizer                                         #
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer    # Used for frequencies of words
import matplotlib.pyplot as plt                                                 # Used for visualisation
from wordcloud import WordCloud                                                 # Used for visualisation
import seaborn as sns                                                           # Seaborn used for barplots
from sklearn.decomposition import LatentDirichletAllocation                     # LDA for topic modeling
import numpy as np 
import umap                                                                     # Used for visulaisation

#%% Readning in the files manually. Ris files

ris_files = [
    Path("IEEE_Xplore_search_A.ris"),
    Path("IEEE_Xplore_search_B.ris"),
    Path("IEEE_Xplore_SearchA_V2.ris"),
    Path("IEEE_Xplore_SearchB_V2.ris"),
    Path("IEEE_Xplore_SearchC_V2.ris"),
    Path("ScienceDirect_search_A.ris"),
    Path("ScienceDirect_SearchA_V2.ris"),
    Path("ScienceDirect_search_B.ris"),
    Path("ScienceDirect_SearchB_V2.ris"),
    Path("ScienceDirect_search_C.ris"),
    Path("ScienceDirect_SearchC_V2.ris")
    ]

all_entries=[]

for fp in ris_files:
    if fp.exists():
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            entries = rispy.load(f)
            all_entries.extend(entries)
            
    else:
        print(f"missing files: {fp}")

print(f"Loaded {len(all_entries)} entries total")

df = pd.DataFrame(all_entries)

#%%
# Standardize title column (IEEE vs ScienceDirect uses different columns for the titles)
df['title'] = df['title'].combine_first(df.get('primary_title', pd.Series())).fillna('')

# Creating normalized title
df['title_norm'] = (
    df['title']
    .astype(str)
    .str.lower()                                       # Lowecasing words
    .str.strip()                                       # Strip
    .str.replace(r'[^\w\s-]', '', regex=True)          # remove punctuation
    .str.replace(r'\s+', ' ', regex=True)              # collapse multiple spaces
)

# Prepare DOI for deduplication (normalize case + whitespace)
df['doi_norm'] = df['doi'].astype(str).str.strip().str.lower()

# Deduplicate primarily on DOI
df = df.drop_duplicates(subset=['doi_norm'], keep='first')

# This is optional, but check if any duplicate DOIs were actually removed
dup_dois = df['doi_norm'].value_counts()
if (dup_dois > 1).any():
    print("\nWarning: duplicate DOIs still present after dedup:")
    print(dup_dois[dup_dois > 1])
else:
    print("No duplicate DOIs remaining.")

# Report from the deduplication results
print(f"After DOI-based deduplication: {len(df)} rows")
print(f"Removed {len(all_entries) - len(df)} entries")

# checking for near-duplicate titles 
dup_titles = df['title_norm'].value_counts()
if (dup_titles > 1).any():
    print("\nSome records have identical normalized titles (but different DOIs):")
    print(dup_titles[dup_titles > 1].head(10))
else:
    print("No identical normalized titles remaining.")


# Convert lists to strings (authors & keywords)
if 'authors' in df.columns:
    df['authors'] = df['authors'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else (x if pd.notna(x) else '')
    )

if 'keywords' in df.columns:
    df['keywords'] = df['keywords'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else (x if pd.notna(x) else '')
    )

# Final selection of columns
columns_final = ['title', 'title_norm', 'authors', 'keywords', 'abstract', 'year', 'doi', 'doi_norm']
df_clean = df[columns_final].copy()


#%% Text cleaning process

#Creating stop words downloading from nltk
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Download WordNet data and POS tagger
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = WordNetLemmatizer()

# Creating a def to clean the text.
def clean_abstract(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Setting the text to lowercase
    text = text.lower()
    
    # Remove copyright and boilerplate
    text = re.sub(r'©.*?(elsevier|ieee|all rights reserved).*?$', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'published by.*', '', text, flags=re.IGNORECASE)
    
    # Remove punctuation, numbers, extra spaces
    text = re.sub(r'[^a-z\s-]', ' ', text)  # keep letters, spaces, hyphens
    text = re.sub(r'\s+', ' ', text).strip()
    
    
    # Splitting the words
    words = text.split()
    
    # Lemmatizing the words
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    # Remove stopwords in text
    cleaned_words = [w for w in lemmatized_words if w not in stop_words and len(w) > 2]
    
    return ' '.join(cleaned_words)

# Apply cleaning of text to abstracts
df_clean['abstract_clean'] = df_clean['abstract'].apply(clean_abstract)

# Preview before/after
print("Before vs After Cleaning")
for i in range(5):
    print(f"Row {i}:")
    print("Raw:   ", df_clean['abstract'].iloc[i][:150] + "..." if len(df_clean['abstract'].iloc[i]) > 150 else df_clean['abstract'].iloc[i])
    print("Clean: ", df_clean['abstract_clean'].iloc[i][:150] + "..." if len(df_clean['abstract_clean'].iloc[i]) > 150 else df_clean['abstract_clean'].iloc[i])
    print("-" * 80)
    
#%% Visualisation of WordCloud, with no filters applied on CountVectorizer
vectorizer_count = CountVectorizer()
X_count = vectorizer_count.fit_transform(df_clean['abstract_clean'])

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
#%% Visualisation of bigrams, unigrams, and comparison for CountVectorier and TF-IDF

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
visualize_nlp_results(df_clean, 'abstract_clean', mode='count', n_range=(1,1), title="Top Unigrams")

# Bigram Counts
visualize_nlp_results(df_clean, 'abstract_clean', mode='count', n_range=(2,2), title="Top Bigrams")

# Unigram TF-IDF
visualize_nlp_results(df_clean, 'abstract_clean', mode='tfidf', n_range=(1,1), title="TF-IDF Unigrams")

# Bigram TF-IDF
visualize_nlp_results(df_clean, 'abstract_clean', mode='tfidf', n_range=(2,2), title="TF-IDF Bigrams")

# Bigram Comparison
visualize_nlp_results(df_clean, 'abstract_clean', mode='compare', n_range=(2,2), top_n=10, title="Bigram Context")


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
    'machine learning': 'machine_learning',
    'condition monitoring': 'condition_monitoring',
    'predictive maintenance': 'predictive_maintenance',
    'fault detection': 'fault_detection',
    'fault diagnosis': 'fault_diagnosis',
    'acoustic signals': 'acoustic_signals',
    'cnn lstm': 'cnn_lstm',
    'renewable energy': 'renewable_energy',
    'hydro turbine': 'hydro_turbine',
    'power plants': 'power_plants',
    'power plant': 'power_plant',
    'hydropower plants': 'hydropower_plants',
    'hydropower plant': 'hydropower_plant',
    'deep learning': 'deep_learning',
    'neural network': 'neural_network',
    'anomaly detection': 'anomaly_detection',
    'real time': 'real_time',
    'neural network': 'neural_network',
    'hydropower unit': 'hydropower_unit'
}

# Manually setting domain specific stop words from TF-IDF, CountVectoriser found earlier
domain_specific_stops = {
    'data', 'time', 'based', 'method', 'methods', 'model', 'models',
    'using', 'paper', 'proposed', 'study', 'results', 'result', 'show', 'shown',
    'experimental', 'presents', 'proposes', 'proposed', 'real', 'approach',
    'performance', 'case', 'cases', 'present', 'presented', 'therefore',
    'however', 'thus', 'respectively', 'accuracy', 'analysis', 'generation',
    'operation', 'operations', 'implementation', 'framework', 'research', 'researches'
    #Legger til tilleg, ser at ofte så kommer feature, og prediction. Kan disse fjernes?
    # More can be added if needed
}

# Apply to the DataFrame before tokenization
df_clean['abstract_clean'] = df_clean['abstract_clean'].apply(lambda x: merge_phrases(x, multi_word_phrases))

df_clean['abstract_clean'] = df_clean['abstract_clean'].apply(second_clean)

# Using Gensim dictionary for coherence
texts = df_clean['abstract_clean'].str.split().tolist()

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
    'Topics': [],                       # Total number of themes the model is to find
    'Alpha': [],                        # Document - topic density
    'Beta': [],                         # Topic-word density
    'min_df': [],                       # Minimum document frequency for a word to be included
    'max_df': [],                       # Maximum document frequency
    'Top_N_Words': [],                  # Number of words used to represent each topic
    'Coherence': [],                    # Logic score
    'Perplexity': [],                   # Statistical score
    'Vocab_Size': [],                   # Total uniqe words used after filtering
    'Top_words_per_topic': []           # Actual keywords that defines each discovered theme
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
                        for number_words in number_of_words:
                        
                            vectorizer = CountVectorizer(
                                min_df = min_df,
                                max_df = max_df,
                                stop_words = None
                                )
                            
                            X = vectorizer.fit_transform(df_clean['abstract_clean'])
                            
                            lda = LatentDirichletAllocation(
                                
                                n_components=n_topics,
                                doc_topic_prior=alpha,
                                topic_word_prior=beta,
                                max_iter=30,
                                learning_method='batch',
                                random_state=42,
                                n_jobs=-1
                                )
                            
                            lda.fit(X)
                            
                            coherence, topics = compute_coherence(
                                lda, 
                                vectorizer.get_feature_names_out(),
                                top_n=number_words)
                            
                            perplexity = lda.perplexity(X)
                            
                            model_results['Topics'].append(n_topics)
                            model_results['Alpha'].append(alpha if alpha is not None else 'symmetric')
                            model_results['Beta'].append(beta if beta is not None else 'symmetric')
                            model_results['min_df'].append(min_df)
                            model_results['max_df'].append(max_df)
                            model_results['Top_N_Words'].append(number_words)
                            model_results['Perplexity'].append(perplexity)
                            model_results['Vocab_Size'].append(X.shape[1])
                            model_results['Coherence'].append(coherence)
                            
                            # Convert topics to string representation
                            #topic_strings = [" ".join(words) for words in topics]
                            #all_topics_str = " | ".join(topic_strings)
                            model_results['Top_words_per_topic'].append(topics)
                            
                            pbar.update(1)

# Saving the results     
df_results = pd.DataFrame(model_results)

#%% Saving the results to a CSV file

df_results.to_csv('lda_model_abstract_results_UMASS_final.csv', sep=';', index=False)

print("\nTop 10 models by highest coherence:")
print(df_results.sort_values('Coherence', ascending=False).head(10))

#%% Visualizing coherence scores across alphas and betas, creating a faceit grid showing all combinations

# Filter for all desired word counts at once
df_facet = df_results[df_results['Top_N_Words'].isin(number_of_words)]

# Add hue to FacetGrid to separate the 4 lines by word count
g = sns.FacetGrid(df_facet, col="Beta", row="Alpha", hue="Top_N_Words", margin_titles=True)

# Map the lineplot 
g.map(sns.lineplot, "Topics", "Coherence", marker="o")
g.map(plt.grid, color='lightgrey', linestyle='--')

# Adding a legend for number of words
g.add_legend(title="Word Combinations", adjust_subtitles=True)
sns.move_legend(g, loc="upper left", frameon=True)
                
# Setting limits on axis and styling the plot
g.set(ylim=(-4, 0.1), yticks=np.arange(-4, 0.1, 0.4))
g.set_axis_labels("Topics", "CV")
g.fig.suptitle("Coherence Peaks across Alpha/Beta & Word Counts", fontsize=13, y=0.97)
g.fig.tight_layout(rect=[0, 0.02, 1, 0.92])

plt.show()

#%% Plotting the top words from the final LDA model

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(1, 5, figsize=(30, 15), sharex=True)
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
X_lda = vectorizer_lda.fit_transform(df_clean['abstract_clean'])

n_topics = 5  # Change here for the number of topics

lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=30,
    learning_method='batch',
    random_state=42,
    n_jobs=-1,
    doc_topic_prior=0.5,
    topic_word_prior=0.5
)

lda.fit(X_lda)

def display_topics(model, feature_names, n_top_words=4):
    for i, topic in enumerate(model.components_):
        print(f"Topic {i+1}: {' '.join([feature_names[j] for j in topic.argsort()[-n_top_words:][::-1]])}")

print("\n=== LDA Topics (top 10 words each) ===\n")
display_topics(lda, vectorizer_lda.get_feature_names_out())


tf_feature_names = vectorizer_lda.get_feature_names_out()
plot_top_words(lda, tf_feature_names, 4, title="Topics in LDA model")

#%% Creating UMAPS for different K neighbours, 5,10,15,20


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
    n_neighbors=5,          # ditt beste valg
    min_dist=0.1,
    random_state=42
)
doc_2d = reducer.fit_transform(doc_topics)

# Dine topic-navn (endre til dine endelige)
topic_labels = [
    "LSTM fault-pred.",           # Topic 0
    "Turbine monitoring",         # Topic 1
    "Stator maintenance",         # Topic 2
    "Signal fault det.",          # Topic 3
    "Hydropower"                  # Topic 4
]

# Plot
plt.figure(figsize=(8, 6))  # litt mindre figur siden legend er inni

dominant = np.argmax(doc_topics, axis=1)
scatter = plt.scatter(
    doc_2d[:, 0], doc_2d[:, 1],
    c=dominant,
    cmap='viridis',
    s=50,
    alpha=0.7,
    edgecolor='none'
)

# Legend inni plottet
cbar = plt.colorbar(scatter, ticks=range(5), orientation='vertical')
cbar.ax.set_yticklabels(topic_labels)  # bruker dine navn!
#cbar.set_label('Dominant Topic')


plt.title('UMAP Projection of Abstracts')
plt.xlabel('UMAP dimension 1')
plt.ylabel('UMAP dimension 2')
plt.grid(True, alpha=0.3)
#plt.tight_layout()
plt.show()
