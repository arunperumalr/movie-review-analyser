# ============================================
# NLP Sentiment Analysis using NLTK + sklearn
# ============================================

# Install required packages (Run once)
# pip install nltk scikit-learn

# Imports NLP tools: movie_reviews: +ve & -ve reviews
import nltk
from nltk.corpus import movie_reviews

# Removing unnecessary words
from nltk.corpus import stopwords
# Splitting text into words
from nltk.tokenize import word_tokenize
# Used to reduce words to root forms: walking-walk | studies-study
from nltk.stem import PorterStemmer, WordNetLemmatizer
# TF-IDF: Converts text into numbers. Machine Learning models cannot understand text directly.
from sklearn.feature_extraction.text import TfidfVectorizer
# Splitting the dataset into training/testing data
from sklearn.model_selection import train_test_split
# Naive Bayes(NB): spam detection | sentiment analysis | text classification
from sklearn.naive_bayes import MultinomialNB
# Measuring how accurate the AI model is
from sklearn.metrics import accuracy_score

# ============================================
# Load Dataset
# ============================================
# empty lists: texts (movie review text) | labels (positive/negative)
texts = []
labels = []

# Loops through ALL movie reviews: Gets text and labels for each fileid
for fileid in movie_reviews.fileids():
    review_text = movie_reviews.raw(fileid)
    label = movie_reviews.categories(fileid)[0]

    # Stores review text : texts = ["Amazing movie", "Worst film"]
    texts.append(review_text)
    # Stores review label : labels = ["pos", "neg"]
    labels.append(label)

print("Total Reviews:", len(texts))

# ============================================
# Sample Review
# ============================================

print("\n===== SAMPLE REVIEWS =====\n")
# Prints first 3 reviews
for i in range(3):
    print("Label:", labels[i])
    print("Review:", texts[i][:300])
    print("-" * 60)

# ============================================
# NLP Preprocessing
# ============================================
# Loads common meaningless words: These are removed(the, is, was, and)
stop_words = set(stopwords.words('english'))
# Reduces words to rough root: playing -> play
stemmer = PorterStemmer()

# More intelligent root conversion: cars -> car
lemmatizer = WordNetLemmatizer()

# ============================================
# Preprocessing Function
# ============================================
# This function cleans ONE review: "This movie was AMAZING!!!" -> "movie amazing"
def preprocess(text, use_stemming=False):

    # Convert to lowercase. Why? Movie and movie means the same
    text = text.lower()

    # Tokenize words: "I love movies" -> ["I", "love", "movies"]
    words = word_tokenize(text)

    # Stores cleaned words.
    clean_words = []

    # Processes one word at a time.
    for word in words:

        # Keep only alphabets(removes punctuation and numbers) and remove stopwords
        if word.isalpha() and word not in stop_words:

            # Stemming
            if use_stemming:
                clean_word = stemmer.stem(word)

            # Lemmatization
            else:
                clean_word = lemmatizer.lemmatize(word)

            # Adds cleaned word.
            clean_words.append(clean_word)

    # ["good", "movie"] -> "good movie"
    return " ".join(clean_words)

# ============================================
# Clean Entire Dataset
# ============================================

print("\nPreprocessing reviews...")
# Processes ALL reviews: Send review one by one to preprocess()
# Its a list: clean_texts = ["movie amazing", "worst film ever"]
clean_texts = [preprocess(text, use_stemming=False) for text in texts]

print("Preprocessing completed!")

# =========================================================
# Convert Text -> Numbers using TF-IDF
# TF-ID:
# Higher importance to meaningful | rare | important words
# Lower Importance to common words
# ==========================================================

# Tells TF-IDF Only keep the 5000 most useful/important words.
# Usually based on word frequency and importance score
vectorizer = TfidfVectorizer(max_features=5000)

# Creates numerical matrix.
X = vectorizer.fit_transform(clean_texts)

y = labels

# (rows, columns) -> (reviews, feature words) -> (No of reviews, 5000)
# since we passed max_features=5000
print("\nTF-IDF Shape:", X.shape)

# ============================================
# Split Dataset: Splits data into:
# Training: Teach model
# Testing: Evaluate model
# 20% for testing.80% for training
# Fixes accuracy changes so accuracy and result
# will not change every run
# Run 1:Review 1 → train, Review 2 → test
# Run 2:Review 1 → test, Review 2 → train
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ============================================
# Train Model
# ============================================

# Creates Naive Bayes classifier.
model = MultinomialNB()

# Training step: AI learns patterns like:
# amazing -> positive | terrible -> negative

model.fit(X_train, y_train)

print("\nModel training completed!")

# ============================================
# Predictions
# ============================================

# Model predicts sentiments for unseen reviews.
y_pred = model.predict(X_test)

# ============================================
# Accuracy: Measures correctness
# Accuracy = 85%: 85 out of 100 predictions correct
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# ============================================
# Test with Custom Reviews
# ============================================

print("\n===== CUSTOM PREDICTIONS =====\n")

# ============================================
# Read Reviews Using Separator
# ============================================

with open("data/movie_reviews.txt", "r", encoding="utf-8") as file:

    content = file.read()

custom_reviews = [
    review.strip()
    for review in content.split("===REVIEW===")
    if review.strip()
]

for review in custom_reviews:

    # Preprocess
    cleaned = preprocess(review)

    # Convert to TF-IDF
    vector = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(vector)[0]

    print("Review:", review)
    print("Prediction:", prediction)
    print("-" * 60)