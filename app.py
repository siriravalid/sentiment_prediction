import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
model = load_model('sentiment_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
st.title('Sentiment Analysis App- Siri')
st.write('Enter a text to get sentiment prediction.')
user_input = st.text_area("Review Text")
if st.button('Predict'):
    if user_input:
        input_seq = tokenizer.texts_to_sequences([user_input])
        input_pad = pad_sequences(input_seq, maxlen=200)
        prediction = model.predict(input_pad)
        sentiment = 'Positive' if prediction[0][1] > prediction[0][0] else 'Negative'
        st.write(f'Sentiment: {sentiment}')
    else:
        st.write("Please enter for a review.")
