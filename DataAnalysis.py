import streamlit as st
from PIL import Image

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

#a59df8
st.markdown(page,unsafe_allow_html=True)
st.title("Next Word Prediction")
def main():

    st.write("Exploratory Data Analysis")
    st.write("1. We will first print the top 20 most common words from our dataset")

    code = '''Top 20 most common words in book1
 
 [('the', 2153), ('a', 1339), ('to', 1256), ('and', 1221), ('I', 958), ('of', 850), ('you', 713), ('was', 638), ('in', 602), ('Harry', 550), ('he', 521), ('it', 483), ('his', 444), ('that', 398), ('on', 371), ('had', 353), ('at', 349), ('for', 325), ('with', 308)]'''
    st.code(code, language='python')

    st.write(
            "2. WordCloud: A word cloud is a collection, or cluster, of words depicted in different sizes. The bigger and bolder the word appears, the more often it’s mentioned within a given text and the more important it is."
            "Word clouds are great for visualizing unstructured text data and getting insights on trends and patterns.")

    image = Image.open('hhp.jpg')
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

['if could go anywher vacat , would go ?', 'i like rainforest , i know requir extens train beforehand i heard rainforest southeast asia ziplin tree tree .', 'when younger , harri dream unknown relat come take away , never happen ; dursley famili .']'''
    st.code(code, language='python')

    st.write("4. Lemmatization: The process of reducing the different forms of a word to one single form. The purpose of lemmatization is same as that of stemming but overcomes the drawbacks of stemming. In stemming, for some words, it may not give may not give meaningful representation. Here, lemmatization comes into picture as it gives meaningful word.")
    code = '''Result after Lemmatization

['If could go anywhere vacation , would go ?', 'I like rainforest , I know requires extensive training beforehand I heard rainforest southeast Asia zipline tree tree .', 'When younger , Harry dreamed unknown relation coming take away , never happened ; Dursleys family .']'''
    st.code(code, language='python')

    st.write("5. BagOfWords: A bag of words is a representation of text that describes the occurrence of words within a document. We just keep track of word counts and disregard the grammatical details and the word order. "
            "One of the biggest problems with text is that it is messy and unstructured, and machine learning algorithms prefer structured, which can be achieved by using the Bag-of-Words technique.")
    code = '''text1 = ['Human Conversation training data',
          'Harry Potter and the Philosophers Stone']
      
Bag Of Words

['conversation' 'data' 'harry' 'human' 'philosophers' 'potter' 'stone'
 'training']
Human Conversation training data
[1 1 0 1 0 0 0 1]
Harry Potter and the Philosophers Stone
[0 0 1 0 1 1 1 0]'''
    st.code(code, language='python')

if __name__ == '__main__':
    main()