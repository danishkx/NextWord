import tensorflow
from tensorflow import keras
from keras.models import load_model
import numpy as np
import pickle
import streamlit as st


page = """
<style>
[data-testid="stSidebar"]{

background-image: url(https://miro.medium.com/max/1400/1*U0hBA8NnTD6l7L8Z2j-eiQ.jpeg);
background-color: rgba(0, 0, 0, 0.5);
opacity: 0.5;
background-size: cover;
}
</style>
"""

st.markdown(page,unsafe_allow_html=True)
st.title("Next Word Prediction")

model = load_model('C:/Users/Admin/Documents/Project/ProjectFinal/Model and Tokenizer/nextword38.h5')
tokenizer = pickle.load(open('C:/Users/Admin/Documents/Project/ProjectFinal/Model and Tokenizer/tokenizer38.pkl', 'rb'))

st.write("Long Short-Term Memory")
text = st.text_input("Enter your line: ")
text1=text

if st.button("Predict"):
    text = text.split(" ")
    text = text[-3:]
    predicted_words = []
    for i in range(3):
        sequence = tokenizer.texts_to_sequences([text])
        sequence = np.array(sequence)
        preds = np.argmax(model.predict(sequence))
        predicted_word = ""
        for key, value in tokenizer.word_index.items():
            if value == preds:
                predicted_word = key
                break
        predicted_words.append(predicted_word)
        text.append(predicted_word)
        text = text[1:]
    st.write(text1 + " " + " ".join(predicted_words))

if st.button("Clear"):
    text = " "
    st.empty()
