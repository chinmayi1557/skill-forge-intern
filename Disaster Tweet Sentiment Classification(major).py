import pandas as pd
import numpy as np
import string
import re

# NLP and ML libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')

# 1. Read dataset
df = pd.read_csv('disaster_tweets_data(DS).csv')

# 2. Handle null values
df.dropna(inplace=True)

# 3. Preprocess tweets
def preprocess(text):
    # a) Tokenize and convert to lower case
    text = text.lower()
    
    # b) Remove URLs and mentions
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)

    # c) Remove punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))

    # d) Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # e) Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

df['clean_tweet'] = df['tweets'].apply(preprocess)

# 4. Transform words into vectors (Choose one: CountVectorizer or TfidfVectorizer)
# You can switch between the two below
vectorizer = TfidfVectorizer()  # or CountVectorizer()
X = vectorizer.fit_transform(df['clean_tweet'])

# 5. Set target variable
y = df['target']

# 6. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train and evaluate models

# a) Multinomial Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# b) Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# c) KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# 9. Evaluate models
def evaluate_model(name, y_test, y_pred):
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

evaluate_model("Multinomial Naive Bayes", y_test, y_pred_nb)
evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("KNN Classifier", y_test, y_pred_knn)

# 10. Find best model
accuracies = {
    "Naive Bayes": accuracy_score(y_test, y_pred_nb),
    "Logistic Regression": accuracy_score(y_test, y_pred_lr),
    "KNN": accuracy_score(y_test, y_pred_knn)
}
best_model = max(accuracies, key=accuracies.get)
print(f"\nBest Performing Model: {best_model} with Accuracy = {accuracies[best_model]:.4f}")
