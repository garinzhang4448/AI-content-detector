import streamlit as st
import pandas as pd
import re
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('preprocessed_mixed_data.csv')

X = df['mixanswer']
y = df['label']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)


tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


lr_model = LogisticRegression()

lr_model.fit(X_train_tfidf, y_train)

y_val_pred = lr_model.predict(X_val_tfidf)

accuracy = accuracy_score(y_val, y_val_pred)

content = st.text_input('Enter your content')
content_tfidf = tfidf_vectorizer.transform([content])

if st.button('Submit'):
    content_tfidf = tfidf_vectorizer.transform([content])
    probability = lr_model.predict_proba(content_tfidf)[:, 0]
    st.write("Probability of Human-generated Answer : {:.2f}%".format(float(probability[0]) * 100))