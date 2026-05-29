# NLP Sentiment Analysis using NLTK and Scikit-learn

A beginner-friendly Natural Language Processing (NLP) project that classifies movie reviews as **Positive** or **Negative** using Python, NLTK, TF-IDF Vectorization, and Multinomial Naive Bayes.

The project trains a machine learning model using the built-in movie reviews dataset from NLTK and then predicts the sentiment of custom reviews loaded from a text file.

---

# Features

* NLP preprocessing pipeline
* Tokenization
* Stopword removal
* Stemming / Lemmatization
* TF-IDF Vectorization
* Naive Bayes Classification
* Custom review prediction from text file
* Beginner-friendly code with detailed comments

---

# Project Structure

```text
movie-review-analyser/
│
├── data/
│   └── movie_reviews.txt
│
├── nlp_sentiment_analysis.py
├── setup_nltk.py
└── README.md
```

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/arunperumalr/movie-review-analyser.git
cd movie-review-analyser
```

---

## 2. Create Virtual Environment (Optional)

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install nltk scikit-learn
```

---

# Download NLTK Resources

Run the setup file once:

```bash
python setup_nltk.py
```

This downloads:

* movie_reviews
* punkt
* punkt_tab
* stopwords
* wordnet

---

# Run the Project

```bash
python nlp_sentiment_analysis.py
```

---

# How It Works

```text
Movie Reviews
      ↓
Text Preprocessing
(lowercase + tokenize + remove stopwords)
      ↓
Lemmatization
      ↓
TF-IDF Vectorization
      ↓
Naive Bayes Model
      ↓
Positive / Negative Prediction
```

---

# Example Output

```text
Model Accuracy: 85.67 %

===== CUSTOM PREDICTIONS =====

Prediction: pos
Prediction: neg
```

---

# Custom Reviews

Custom reviews are loaded from:

```text
data/movie_reviews.txt
```

Reviews are separated using:

```text
===REVIEW===
```

Example:

```text
===REVIEW===

This movie was fantastic and visually stunning.

===REVIEW===

The screenplay was boring and predictable.
```

---

# Technologies Used

* Python
* NLTK
* Scikit-learn
* TF-IDF
* Multinomial Naive Bayes

---

# Learning Goals

This project was built to understand:

* Natural Language Processing (NLP)
* Text preprocessing
* Feature extraction
* Traditional Machine Learning for text classification
* Sentiment analysis

---

# Author

Arun
