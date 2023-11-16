import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import requests
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

# Load the trained models
def load_model():
    model = tf.keras.models.load_model('CNN_models_update.h5')
    word2vec_model = Word2Vec.load('word2vec_model.gensim')  # 确保文件名和路径正确

    # Load the tokenizer from pickle file
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    return model, word2vec_model, tokenizer

model, word2vec_model, tokenizer = load_model()

def text_to_word2vec(text, model):
    words = text.split()
    text_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(text_vectors, axis=0) if text_vectors else np.zeros(model.vector_size)

st.title("AI-content detector application")

input_text = st.text_area("Enter your content")

if st.button("Submit"):
    # CNN prediction part
    sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequences = pad_sequences(sequences, maxlen=5000)  
    word2vec_input = np.array([text_to_word2vec(input_text, word2vec_model)])[..., np.newaxis]
    prediction = model.predict([padded_sequences, word2vec_input])
    prob_cnn = float(prediction[0][0])

    # Sapling part
    response = requests.post(
        "https://api.sapling.ai/api/v1/aidetect",
        json={
            "key": "RXE0ATFO4HO106O1AQHDO0FKL3ZAOJSZ", 
            "text": input_text
        })
    content_by_sapling = response.json()
    prob_sapling = content_by_sapling['score']

    st.write("The probability of Human-generated answer using CNN model: {:.2f}%".format((1-prob_cnn)*100))
    st.write("The Probability of Human-generated Answer using Sapling: {:.2f}%".format((1-float(prob_sapling)) * 100))
