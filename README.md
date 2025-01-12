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
