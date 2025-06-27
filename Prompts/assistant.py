from langchain_groq import ChatGroq
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatGroq(model='llama-3.1-8b-instant')

st.header('Machine Learning Assistant')

model_input = st.selectbox('Choose the Concept which you want to study', [
    'Regression', 'Classification', 'Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'Random Forest', 'PCA', 'K-Means',
    'Gradient Descent', 'Multi Layer Perceptron', 'Convolutional Neural Network', 'Recurrent Neural Network', 'LSTM', 'GRU', 'Transformers',
    'BERT', 'GPT', 'T5', 'Autoencoders', 'GANs', 'Reinforcement Learning'
])

style_input = st.selectbox('Enter the type of Explanation you want', [
    'Beginner-Friendly', 'Technical', 'Code-Oriented', 'Mathematical', '5-year old'
])

length_input = st.selectbox('Select Explanation Length',[
    'Short(1-2 Paragraphs)', 'Medium(3-5 Paragraphs)', 'Long(Detailed-Explanation)'
])

template = load_prompt('template.json')

prompt = template.format(
    topic=model_input,
    style=style_input,
    length=length_input
)

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)
    st.success('Assistant work succesfully')