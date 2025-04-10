import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load the LSTM Model
model = load_model('next_word_lstm.h5')
## lad the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    #convert the text to sequence
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):] #ensure the sequence length matches the max_sequence length
    #pad the sequence
    token_list=pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    #predict the next word
    predicted=model.predict(token_list, verbose=0)
    #get the index of the predicted word
    #argmax return the index word of highest probability
    predicted_word_index=np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None 

# Streamlit app
st.title("Next Word Prediction with LSTM")
input_text = st.text_input("Enter the sequence of words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence =  model.input_shape[1]+1 #Retrieve the max sequence length
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence)
    st.write(f"The next word is: {next_word}")