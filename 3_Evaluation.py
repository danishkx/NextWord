
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

#a59df8
st.markdown(page,unsafe_allow_html=True)

st.title("Next Word Prediction")
st.write("Evaluation of LSTM and GRU Model")
st.write("1. Precision: It is the ratio of the number of true positive predictions to the number of "
         "true positive plus false positive predictions. It measures the proportion of positive "
         "predictions that are actually correct.Precision should ideally be 1 (high) for a good classifier")
code = '''LSTM Model
Precision: 87.98%
GRU Model
Precision: 87.87%'''
st.code(code, language='python')

st.write("2. Recall: It is the ratio of the number of true positive predictions to the number of true positive plus "
         "false negative predictions. It measures the proportion of actual positive instances that are correctly predicted."
         "Recall should ideally be 1 (high) for a good classifier")
code = '''LSTM Model
Recall: 88.25%
GRU Model
Recall: 87.98%'''
st.code(code, language='python')

st.write("3. F1-score: It is the harmonic mean of precision and recall. It balances both precision and recall "
         "and provides a single score for the overall performance of the model.")
code = '''LSTM Model
F1-score: 87.43%
GRU Model
F1-score: 87.32%'''
st.code(code, language='python')

st.write("4. Accuracy: It is the ratio of the number of correct predictions to the total number of predictions. "
         "It measures the overall performance of the model.")
code = '''LSTM Model
Accuracy: 92.34%
GRU Model
Accuracy: 92.85%'''
st.code(code, language='python')

st.write("5. Epoch vs Loss Graph: The epoch vs loss graph shows how the loss changes over the course of training, which can provide valuable insights into the model's behavior.")
st.image("finallstm.jpg")

st.write("6. Epoch vs Accuracy Graph: The epoch vs accuracy graph shows how the accuracy changes over the course of training, which can provide valuable insights into the model's performance.")
st.image("finallstmacc.jpg")

st.write("These metrics are usually used together to understand the performance of a model in different ways. "
         "They provide a comprehensive evaluation of the model's performance and can be used to compare different models.")

