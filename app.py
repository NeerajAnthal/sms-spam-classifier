import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import sklearn
nltk.download('punkt')

ps =PorterStemmer()
tf =pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))
st.title("Email/SMS Spam Classifier")
input_sms=st.text_area("Enter the message")




def transforming_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation :
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

if st.button('Predict'):
    # Preprocessing
    transform_sms=transforming_text(input_sms)

    # Vectorizing
    vector_input=tf.transform([transform_sms])
    # Predict
    result = model.predict(vector_input)[0]

    # Display
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")