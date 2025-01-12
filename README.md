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
