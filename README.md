# NLP Examples with NLTK

This section demonstrates basic NLP tasks using the NLTK library. The example includes tokenization, removing stopwords, and part-of-speech tagging.

---

## Code 1: Tokenization, Stopwords Removal, and POS Tagging with NLTK

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

# Input text
text = "Natural Language Processing in an exciting field of Artificial Intelligence"

# Get the list of stop words in English
stop_words = set(stopwords.words("english"))

# Tokenize the sentence
tokens = word_tokenize(text)

# Remove stop words
filter_words = []
for word in tokens:
    if word.lower() not in stop_words:
        filter_words.append(word)

# Perform Part-of-Speech tagging
pos_tags = nltk.pos_tag(filter_words)

# Display results
print("Original Tokens:", tokens)
print("Filtered Tokens (without stop words):", filter_words)
print("POS Tags:", pos_tags)



# NLP Examples with TF-IDF

This section demonstrates how to compute the Term Frequency-Inverse Document Frequency (TF-IDF) for a set of documents using the `TfidfVectorizer` from `sklearn`.

---

## Code 2: TF-IDF with Scikit-Learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
docs = [
    "Natural Language processing is amazing.",
    "Machine learning and NLP go hand in hand.",
    "TF-IDF helps find important words in a document."
]

# Initialize the TfidfVectorizer
vec = TfidfVectorizer()

# Fit and transform the documents into a TF-IDF matrix
tfidf_mat = vec.fit_transform(docs)

# Display the feature names (words)
print(vec.get_feature_names_out())

# Display the TF-IDF matrix as an array
print(tfidf_mat.toarray())

