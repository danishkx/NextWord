import tensorflow
from tensorflow import keras
from keras.models import load_model
import numpy as np
import pickle
import streamlit as st
from PIL import Image

st.title("Next Word Prediction")


def main():
    model = load_model('C:/Users/Admin/Documents/Project/modelnew/nextword40.h5')
    tokenizer = pickle.load(open('C:/Users/Admin/Documents/Project/modelnew/tokenizer40.pkl', 'rb'))
    model1 = load_model('C:/Users/Admin/Documents/Project/modelnew/nextword36.h5')
    tokenizer1 = pickle.load(open('C:/Users/Admin/Documents/Project/modelnew/tokenizer36.pkl', 'rb'))
    nav = st.sidebar.radio("Navigation", ["EDA", "LSTM Model", "GRU Model","Evaluation"])

    if nav == "EDA":
        st.write("Exploratory Data Analysis")
        st.write("Datasets")
        st.write("The first dataset I have used is a human conversation corpus which I found on Kaggle. "
                 "This dataset contains 19,546 words and the filesize is 105 KB. "
                 "Here is the link to access the dataset, https://www.kaggle.com/datasets/projjal1/human-conversation-training-data")
        st.write("The second dataset I have used is a Harry Potter book corpus which I found on Kaggle. "
         "This dataset contains 37,185 words and the filesize is 216 KB. "
         "Here is the link to access the dataset, https://www.kaggle.com/datasets/balabaskar/harry-potter-books-corpora-part-1-7")
        st.write("The third dataset I have used is a SeaFood recipe book corpus which I found on PDFDrive. "
         "This dataset contains 55,938 words and the filesize is 313 KB. "
         "Here is the link to access the dataset, https://www.pdfdrive.com/seafood-recipes-e34780763.html")
        st.write("We will now perform data analysis on all the datasets.")
        st.write("1. We will first print the top 20 most common words from our dataset")

        code = '''Top 20 most common words in book1 
        
[('the', 3791), ('and', 3139), ('a', 2173), ('to', 2080), ('1', 1803), ('in', 1465), ('of', 1462), ('with', 985), ('I', 978), ('you', 798), ('cup', 774), ('for', 725), ('2', 719), ('on', 690), ('was', 649), ('teaspoon', 642), ('it', 626), ('1/2', 596), ('tablespoon', 591), ('or', 550)]'''
        st.code(code, language='python')

        st.write(
            "2. WordCloud: A word cloud is a collection, or cluster, of words depicted in different sizes. The bigger and bolder the word appears, the more often it’s mentioned within a given text and the more important it is."
            "Word clouds are great for visualizing unstructured text data and getting insights on trends and patterns.")

        image = Image.open('download.jpg')
        st.image(image)

        st.write(
            "3. Stemming: Stemming is a natural language processing technique that is used to reduce words to their base form, also known as the root form. The process of stemming is used to normalize text and make it easier to process ")
        code = '''paragraph1 = """If you could go anywhere on vacation, where would you go?
I like rainforest, but I know it requires extensive training beforehand 
I heard there are rainforests in southeast Asia where you can zipline from tree to tree.
            
When he had been younger, Harry had dreamed of some unknown relation coming to take 
him away, but it had never happened; the Dursleys were his only family.
            
Place fish steaks or fillets in a baking dish and spoon some syrup over. 
Marinate in the refrigerator for 1 hour. Cook the fish on a hot grill, basting with a teaspoon 
of barbecue syrup on each side. """

Result after Stemming 

['if could go anywher vacat , would go ?', 'i like rainforest , i know requir extens train beforehand i heard rainforest southeast asia ziplin tree tree .', 'when younger , harri dream unknown relat come take away , never happen ; dursley famili .', 'place fish steak fillet bake dish spoon syrup .', 'marin refriger 1 hour .', 'cook fish hot grill , bast teaspoon barbecu syrup side .']'''
        st.code(code, language='python')

        st.write("4. Lemmatization: The process of reducing the different forms of a word to one single form. The purpose of lemmatization is same as that of stemming but overcomes the drawbacks of stemming. In stemming, for some words, it may not give may not give meaningful representation. Here, lemmatization comes into picture as it gives meaningful word.")
        code = '''Results after Lemmatization 

['If could go anywhere vacation , would go ?', 'I like rainforest , I know requires extensive training beforehand I heard rainforest southeast Asia zipline tree tree .', 'When younger , Harry dreamed unknown relation coming take away , never happened ; Dursleys family .', 'Place fish steak fillet baking dish spoon syrup .', 'Marinate refrigerator 1 hour .', 'Cook fish hot grill , basting teaspoon barbecue syrup side .']'''
        st.code(code, language='python')

        st.write("5. BagOfWords: A bag of words is a representation of text that describes the occurrence of words within a document. We just keep track of word counts and disregard the grammatical details and the word order. "
            "One of the biggest problems with text is that it is messy and unstructured, and machine learning algorithms prefer structured, well defined fixed-length inputs and by using the Bag-of-Words technique we can convert variable-length texts into a fixed-length vector.")
        code = '''text1 = ['Human Conversation training data',
          'Harry Potter and the Philosophers Stone',
          'Ultimate Collection of Seafood recipes']

Results  
     
['collection' 'conversation' 'data' 'harry' 'human' 'philosophers'
'potter' 'recipes' 'seafood' 'stone' 'training' 'ultimate']
Human Conversation training data
[0 1 1 0 1 0 0 0 0 0 1 0]
Harry Potter and the Philosophers Stone
[0 0 0 1 0 1 1 0 0 1 0 0]
Ultimate Collection of Seafood recipes
[1 0 0 0 0 0 0 1 1 0 0 1]'''
        st.code(code, language='python')

    if nav == "LSTM Model":
        st.write("Long Short-Term Memory")
        text = st.text_input("Enter your line: ")

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
            st.write(" ".join(predicted_words))

        if st.button("Clear"):
            text = " "
            st.text("")

    if nav == "GRU Model":
        st.write("Gated Recurrent Unit")
        text = st.text_input("Enter your line: ")

        if st.button("Predict"):
            text = text.split(" ")
            text = text[-3:]
            predicted_words = []
            for i in range(3):
                sequence = tokenizer1.texts_to_sequences([text])
                sequence = np.array(sequence)
                preds = np.argmax(model1.predict(sequence))
                predicted_word = ""
                for key, value in tokenizer1.word_index.items():
                    if value == preds:
                        predicted_word = key
                        break
                predicted_words.append(predicted_word)
                text.append(predicted_word)
                text = text[1:]
            st.write(" ".join(predicted_words))

        if st.button("Clear"):
            text = " "
            st.empty()
            #st.text("")


    if nav == "Evaluation":
        st.write("Evaluation of LSTM and GRU Model")
        st.write("1. Precision: It is the ratio of the number of true positive predictions to the number of "
"true positive plus false positive predictions. It measures the proportion of positive "
"predictions that are actually correct.Precision should ideally be 1 (high) for a good classifier")
        code = '''LSTM Model
Precision: 0.8647%
GRU Model
Precision: 0.8645%'''
        st.code(code, language='python')

        st.write("2. Recall: It is the ratio of the number of true positive predictions to the number of true positive plus "
"false negative predictions. It measures the proportion of actual positive instances that are correctly predicted."
"Recall should ideally be 1 (high) for a good classifier")
        code = '''LSTM Model
Recall: 0.8627%
GRU Model
Recall: 0.8613%'''
        st.code(code, language='python')
	
        st.write("3. F1-score: It is the harmonic mean of precision and recall. It balances both precision and recall "
                 "and provides a single score for the overall performance of the model.")
        code = '''LSTM Model
F1-score: 0.8561%
GRU Model
F1-score: 0.8548%'''
        st.code(code, language='python')

        st.write("4. Accuracy: It is the ratio of the number of correct predictions to the total number of predictions. "
                 "It measures the overall performance of the model.")
        code = '''LSTM Model
Accuracy: 89.19%
GRU Model
Accuracy: 89.26%'''
        st.code(code, language='python')

        st.write("These metrics are usually used together to understand the performance of a model in different ways. "
             "They provide a comprehensive evaluation of the model's performance and can be used to compare different models.")

if __name__ == '__main__':
    main()