import nltk

# ============================================
# Run Once: Download required datasets
# ============================================

# Downloads movie review dataset.
nltk.download('movie_reviews')
# Downloads tokenizer: "I love NLP" -> ["I", "love", "NLP"]
nltk.download('punkt')
nltk.download('punkt_tab')
# Downloads common useless words: the, is, a, an, on
nltk.download('stopwords')
# Needed for lemmatization.
nltk.download('wordnet')