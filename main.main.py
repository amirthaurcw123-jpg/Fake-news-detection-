# 📰 Fake News Detection System

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 📌 Load dataset
# dataset.csv ல இரண்டு columns இருக்கணும்: "text", "label"
# label: 1 = Fake, 0 = Real

data = pd.read_csv("dataset.csv")

# 📌 Replace missing values
data = data.fillna("")

# 📌 Features & Labels
X = data["text"]
Y = data["label"]

# 📌 Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 📌 Convert text into numbers (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 📌 Train Model
model = LogisticRegression()
model.fit(X_train_tfidf, Y_train)

# 📌 Test model
Y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(Y_test,