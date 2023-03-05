# -*- coding: utf-8 -*-
"""ProjectFinal.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lDBI0-98H3ivCltV0sh85mtF06LxmtsO
"""

from keras.models import load_model
import numpy as np
import pickle

model = load_model('/content/drive/MyDrive/Colab Notebooks/nextword38.h5')
tokenizer = pickle.load(open('/content/drive/MyDrive/Colab Notebooks/tokenizer38.pkl', 'rb'))

while True:
    text = input("Enter your line (type 'quit' to exit): ")
    text1=text
    if text == 'quit':
        break
    text = text.split(" ")
    text = text[-3:]
    predictions = []
    for i in range(3):
        predicted_words = []
        for j in range(3):
            sequence = tokenizer.texts_to_sequences([text])
            sequence = np.array(sequence)
            preds = model.predict(sequence)
            top_prediction = np.argmax(preds[0])
            predicted_word = ""
            for key, value in tokenizer.word_index.items():
                if value == top_prediction:
                    predicted_word = key
                    break
            predicted_words.append(predicted_word)
            text.append(predicted_word)
            text = text[1:]
        predictions.append(predicted_words)
    for i, words in enumerate(predictions):
        print(f"Prediction {i + 1}: {' '.join(words)}")
    selected_prediction = int(input("Please select a prediction (1, 2, or 3): "))
    selected_words = predictions[selected_prediction - 1]
    print("You selected:" + text1 + " " +  ' '.join(selected_words))
    text = text[-3:] + selected_words