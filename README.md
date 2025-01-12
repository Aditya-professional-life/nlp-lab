# NLTK POS Tagging and Stop Words Filtering

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

text = "Natural Language Processing in an exiting feild of Artifical Intelligence"

stop_words = set(stopwords.words("english"))
tokens = word_tokenize(text)

filter_words = [word for word in tokens if word.lower() not in stop_words]

pos_tags = nltk.pos_tag(filter_words)

print("Tokens:", tokens)
print("Filtered Words:", filter_words)
print("POS Tags:", pos_tags)


# TF-IDF Example with Scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "Natural Language processing is amazing.",
    "Machine learning and NLP go hand in hand.",
    "TF-IDF helps find important words in a document."
]

vec = TfidfVectorizer()

tfidf_mat = vec.fit_transform(docs)

print(vec.get_feature_names_out())
print(tfidf_mat.toarray())



# TF-IDF Example with Scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "Natural Language processing is amazing.",
    "Machine learning and NLP go hand in hand.",
    "TF-IDF helps find important words in a document."
]

vec = TfidfVectorizer()

tfidf_mat = vec.fit_transform(docs)

print(vec.get_feature_names_out())
print(tfidf_mat.toarray())
