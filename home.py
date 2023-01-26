import tensorflow
from tensorflow import keras
from keras.models import load_model
import numpy as np
import pickle

model = load_model('C:/Users/Admin/Documents/Project/nextword91.h5')
tokenizer = pickle.load(open('C:/Users/Admin/Documents/Project/tokenizer91.pkl', 'rb'))

def Predict_Next_Words(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence))
    predicted_word = ""

    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break

    print(predicted_word)
    return predicted_word

while (True):

    text = input("Enter your line: ")

    if text == "stop the script":
        print("Ending The Program.....")
        break
    else:
        try:
            text = text.split(" ")
            text = text[-3:]
            print(text)

            Predict_Next_Words(model, tokenizer, text)
        except Exception as e:
            print("Error: ", e)
            continue

# streamlit run C:\Users\Admin\PycharmProjects\nextword\home.py


import streamlit as st
from PIL import Image

st.title("Next Word Prediction")

nav = st.sidebar.radio("Navigation", ["EDA", "LSTM Model", "GRU Model"])

if nav == "EDA":
    st.write("Exploratory Data Analysis")
    st.write("Datasets")
    st.write("The first dataset I have used is a human conversation corpus which I found on Kaggle. "
             "This dataset contains 19,546 words and the filesize is 105 KB. "
             "Here is the link to access the dataset, https://www.kaggle.com/datasets/projjal1/human-conversation-training-data")
    st.write("We will now perform data analysis on it.")
    st.write("1. We will first print the top 20 most common words from our dataset")

    code = '''Top 20 most common words in book1 
 [('I', 763), ('to', 524), ('a', 515), ('the', 496), ('you', 496), ('of', 277), ('it', 239), ('and', 231), ('is', 214), ('in', 195), ('for', 187), ('that', 177), ('have', 167), ('like', 153), ('do', 144), ("I'm", 138), ('are', 137), ('your', 116), ('about', 112), ('but', 107)]'''
    st.code(code, language='python')

    st.write(
        "2. WordCloud: A word cloud is a collection, or cluster, of words depicted in different sizes. The bigger and bolder the word appears, the more often it’s mentioned within a given text and the more important it is."
        "Word clouds are great for visualizing unstructured text data and getting insights on trends and patterns.")
    code = ''' wordcloud = WordCloud(width = 1000, height = 600, background_color = 'black', 
    min_font_size = 10,max_words=2000,collocations=False).generate(book1)

plt.figure(figsize = (12, 12), facecolor = 'lavender')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.tight_layout(pad = 2) 
plt.show()'''
    st.code(code, language='python')
    image = Image.open('human.jpg')
    st.image(image)

    st.write(
        "3. Stemming: Stemming is a natural language processing technique that is used to reduce words to their base form, also known as the root form. The process of stemming is used to normalize text and make it easier to process ")
    code = '''paragraph1 = """
If you could go anywhere on vacation, where would you go?
I like rainforest, but I know it requires extensive training beforehand 
I heard there are rainforests in southeast Asia where you can zipline from tree to tree 
I am afraid I will be scared of doing this 
I won't lie, it sounds scary  I'm scared right now just thinking about it """

Result after Stemming 

['if could go anywher vacat , would go ?', "i like rainforest , i know requir extens train beforehand i heard rainforest southeast asia ziplin tree tree i afraid i scare i wo n't lie , sound scari i 'm scare right think"]'''
    st.code(code, language='python')

    st.write(
        "4. Lemmatization: The process of reducing the different forms of a word to one single form. The purpose of lemmatization is same as that of stemming but overcomes the drawbacks of stemming. In stemming, for some words, it may not give may not give meaningful representation. Here, lemmatization comes into picture as it gives meaningful word.")
    code = '''Results after Lemmatization 

['If could go anywhere vacation , would go ?', "I like rainforest , I know requires extensive training beforehand I heard rainforest southeast Asia zipline tree tree I afraid I scared ) I wo n't lie , sound scary I 'm scared right thinking"]'''
    st.code(code, language='python')

    st.write(
        "5. BagOfWords: A bag of words is a representation of text that describes the occurrence of words within a document. We just keep track of word counts and disregard the grammatical details and the word order. "
        "One of the biggest problems with text is that it is messy and unstructured, and machine learning algorithms prefer structured, well defined fixed-length inputs and by using the Bag-of-Words technique we can convert variable-length texts into a fixed-length vector.")
    code = '''text2 = ['I am going to Hawaii' ,
         'This will be my first time visiting Hawaii',
          'I love Hawaii its a good place to be']

Results       
['going' 'good' 'hawaii' 'love' 'place' 'time' 'visiting']
I am going to Hawaii
[1 0 1 0 0 0 0]
This will be my first time visiting Hawaii
[0 0 1 0 0 1 1]
I love Hawaii its a good place to be
[0 1 1 1 1 0 0]'''
    st.code(code, language='python')

if nav == "LSTM Model":
    st.text("Long Short-Term Memory")
    if st.button("Predict"):
        Predict_Next_Words()

if nav == "GRU Model":
    st.title("Gated Recurrent Unit")