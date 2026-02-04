# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 19:46:25 2026

@author: Sander



"""
#%%
import rispy
from pathlib import Path
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from tabulate import tabulate

#%% Readning in the files manually. Ris files
ris_files = [
    Path("IEEE_Xplore_search_A.ris"),
    Path("IEEE_Xplore_search_B.ris"),
    Path("ScienceDirect_search_A.ris"),
    Path("ScienceDirect_search_B.ris"),
    Path("ScienceDirect_search_C.ris")
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
# Science Direct and IEEXlpore have different columns for titles. Moving title columne from primary title to title.
df['title'] = df['title'].combine_first(df['primary_title']).fillna('')

# The following code checks if there is any duplication in the articles found. This is done by checking the titles of the articles
# Normalizing the title with lowercase and whitespace (using lower and strip)
df['title_norm'] = df['title'].astype(str).str.lower().str.strip()

# Deduplication is only done on normalized title. Using df.drop_duplicates, it removes rows that are identical through the dataframe.
df = df.drop_duplicates(subset=['title_norm'], keep='first')

# Cleaning away the helper column title_norm.
df = df.drop(columns=['title_norm'], errors='ignore')

# Checking how many duplicates were removed
print(f"After deduplication: {len(df)} rows")
print(f"Removed {len(all_entries) - len(df)} entries")

# Checking if any duplicates are left by title column
dup_titles = df['title'].value_counts()
if (dup_titles > 1).any():
    print("\nStill some duplicate titles remaining:")
    print(dup_titles[dup_titles > 1])
else:
    print("No duplicate titles remaining.")

# convert lists to strings for Authors & keywords
if 'authors' in df.columns:
    df['authors'] = df['authors'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else (x if pd.notna(x) else '')
    )

if 'keywords' in df.columns:
    df['keywords'] = df['keywords'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else (x if pd.notna(x) else '')
    )

# Changing Journal_name to journal
df['journal']=df.get('journal_name', 'unknown').astype(str).str.strip().fillna('unknown')

# Final selection for the cleaned of Pandas Frame
columns_final = ['title', 'authors', 'keywords', 'abstract', 'year', 'journal', 'doi']
df_clean = df[columns_final].copy()

"""
# Preview of the cleaned dataframe
print("\n=== Cleaned DataFrame Head (first 10 rows) ===\n")
print(df_clean.head(10).to_string(index=False))

print("\n=== Year Distribution (should match your diagnostic) ===\n")
print(df_clean['year'].value_counts().sort_index())

print(f"\nRows with year == 0: {(df_clean['year'] == 0).sum()} (should be 0 now)")
print(f"Rows with abstract: {(df_clean['abstract'].str.strip() != '').sum()}")
print(f"Total rows: {len(df_clean)}")
"""

#%% Creating stop words, downloading from nltk
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

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
    
    # Remove stopwords in text
    words = text.split()
    cleaned_words = [w for w in words if w not in stop_words and len(w) > 2]
    
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
    
#%% Searching for frequency of words
# Raw word frequency (CountVectorizer)
vectorizer_count = CountVectorizer()
X_count = vectorizer_count.fit_transform(df_clean['abstract_clean'])

word_freq = pd.DataFrame(
    X_count.sum(axis=0).T,
    index=vectorizer_count.get_feature_names_out(),
    columns=['count']
).sort_values('count', ascending=False)

print("\n=== Top 30 most frequent words across all abstracts ===")
print(word_freq.head(30))

# TF-IDF
vectorizer_tfidf = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english')  # ignore words in >85% docs, require in ≥2 docs
X_tfidf = vectorizer_tfidf.fit_transform(df_clean['abstract_clean'])
tfidf_df = pd.DataFrame(
    X_tfidf.mean(axis=0).T,
    index=vectorizer_tfidf.get_feature_names_out(),
    columns=['mean_tfidf']
).sort_values('mean_tfidf', ascending=False)

print("\n=== Top 30 most distinctive/important words (TF-IDF mean) ===")
print(tfidf_df.head(30))

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


#Top 20 TF-IDF terms
top20 = tfidf_df.head(20).reset_index()
plt.figure(figsize=(10, 7))
sns.barplot(data=top20, x='mean_tfidf', y='index', palette='viridis')
plt.title("Top 20 Most Distinctive Words (mean TF-IDF across abstracts)", fontsize=14)
plt.xlabel("Mean TF-IDF Score")
plt.ylabel("Term")
plt.tight_layout()
plt.show()

# Bigrams for two-word phrases
vectorizer_bigram = CountVectorizer(ngram_range=(2, 2), min_df=2)
X_bigram = vectorizer_bigram.fit_transform(df_clean['abstract_clean'])
bigram_freq = pd.DataFrame(
    X_bigram.sum(axis=0).T,
    index=vectorizer_bigram.get_feature_names_out(),
    columns=['count']
).sort_values('count', ascending=False)

print("\n=== Top 20 bigrams (two-word phrases) ===")
print(bigram_freq.head(20))

# Bar plot for bigrams
top20_bigram = bigram_freq.head(20).reset_index()
plt.figure(figsize=(10, 7))
sns.barplot(data=top20_bigram, x='count', y='index', palette='mako')
plt.title("Top 20 Most Frequent Bigrams in Abstracts", fontsize=14)
plt.xlabel("Count")
plt.ylabel("Bigram")
plt.tight_layout()
plt.show()

#%%

# LDA usually works better on raw counts (not TF-IDF)
vectorizer_lda = CountVectorizer(max_df=0.90, min_df=2)  # ignore ultra-common words
X_lda = vectorizer_lda.fit_transform(df_clean['abstract_clean'])

n_topics = 13  # start with 7

lda = LatentDirichletAllocation(
    n_components=n_topics,
    max_iter=25,
    learning_method='online',
    random_state=42,
    n_jobs=-1
)

lda.fit(X_lda)

def display_topics(model, feature_names, n_top_words=12):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[-n_top_words:][::-1]]))
        print()

print("\n=== LDA Topics (top 12 words each) ===\n")
display_topics(lda, vectorizer_lda.get_feature_names_out())

# Which papers belong most to which topic?
doc_topics = lda.transform(X_lda)
df_clean['dominant_topic'] = doc_topics.argmax(axis=1) + 1

print("\nTopic distribution (how many papers per topic):")
print(df_clean['dominant_topic'].value_counts().sort_index())

# Show top 5 papers (by title) for each dominant topic
table_data = []

for topic_num in sorted(df_clean['dominant_topic'].unique()):
    subset = df_clean[df_clean['dominant_topic'] == topic_num]
    count = len(subset)
    
    # Top words (last 8 most representative)
    top_words = ' '.join(vectorizer_lda.get_feature_names_out()[lda.components_[topic_num-1].argsort()[-8:][::-1]])
    
    # Get top 3 titles, truncate to 70 chars (but no ... added)
    top_titles = subset['title'].str[:140].head(3).tolist()
    
    # Join with newline only – first title normal, others prefixed with \n
    if len(top_titles) == 0:
        examples = "(no papers)"
    else:
        examples = top_titles[0]
        if len(top_titles) >= 2:
            examples += "\n" + top_titles[1]
        if len(top_titles) >= 3:
            examples += "\n" + top_titles[2]
    
    table_data.append([topic_num, count, top_words, examples])

headers = ["Topic", "Papers", "Top words (short)", "Example titles (top 3)"]

print("\n" + "="*100)
print("LDA Topic Summary – Representative Papers")
print("-"*100)
print(tabulate(
    table_data,
    headers=headers,
    tablefmt="simple_grid",
    maxcolwidths=[6, 8, 40, 70],   # wider column for titles
    stralign="left"
))
print("="*100)

#%%
# Priority topics for the thesis
priority_topics = [2, 4, 6, 3, 7]  # add/remove as you like

topic_names = {
    1: "Hydropower role in renewable / flexible systems",
    2: "ML / machine learning for fault detection & diagnosis",
    3: "Deep learning (CNN-LSTM) for hydro turbine faults",
    4: "machine learning predictive maintenance",
    5: "Anomaly detection and health assessment in hydro units",
    6: "SCADA / sensor / real-time fault monitoring",
    7: "General energy systems challenges"
}


for topic_num in priority_topics:
    print(f"\n{'=' * 80}")
    print(f"Topic {topic_num} – {len(df_clean[df_clean['dominant_topic'] == topic_num])} papers")
    print(f"Suggested focus: {topic_names.get(topic_num, 'Custom topic')}")
    print("-" * 80)
    
    # Get papers in this topic, sorted by strength of belonging to it
    topic_scores = lda.transform(X_lda)[:, topic_num-1]  # probability for this topic
    df_topic = df_clean.copy()
    df_topic['topic_score'] = topic_scores
    
    top_papers = df_topic[df_topic['dominant_topic'] == topic_num] \
        .sort_values('topic_score', ascending=False) \
        .head(5) \
        [['title', 'year', 'journal', 'topic_score', 'doi']]
    
    # Truncate long titles
    top_papers['title'] = top_papers['title'].str[:100] + '...'
    
    print(top_papers.to_string(index=False))
    print(f"Topic score = how strongly the paper belongs to this theme (0–1)")