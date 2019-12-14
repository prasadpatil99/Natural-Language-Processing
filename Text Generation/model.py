# Import libraries
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
import numpy as np 
#import tensorflow.keras.utils as ku 

tokenizer = Tokenizer()

data = open('data.txt').read()

# Preprocessing
corpus = data.lower().split("\n")
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# Creating integer tokens 
input_sequences = []
for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0] 
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)

# Finding maximum integer tokens
max_sequence_len = max([len(x) for x in input_sequences])

# Transform data with highest length
# (padding = 'pre') that is to add seq. before lowest n_gram
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding = 'pre'))

# Create present and future values as features
values = input_sequences[:,:-1]
future_values = input_sequences[:,-1]

# Can also be converted into categorical form
#future_values = ku.to_categorical(label, num_classes=total_words)

# Building LSTM model
# embedding layer turns positive indexes into dense vectors
model = Sequential()
model.add(Embedding(total_words, 70, input_length=max_sequence_len-1)) 
model.add(LSTM(150, return_sequences = True)) # return seq. for input to next layer
model.add(Dropout(0.1))
model.add(LSTM(110))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(values, future_values, epochs=100, verbose=1) 

def prediction(pred_text, next_words, max_sequence_len):
    output = ""
    for _ in range(next_words):
    	token_list = tokenizer.texts_to_sequences([pred_text])[0]
    	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    	predicted = model.predict_classes(token_list)

    	for word, index in tokenizer.word_index.items():
    		if index == predicted:
    			output = word
    			break
    	pred_text += " " + output
    return pred_text

# Predict output from we..
print(prediction("I", 3 , max_sequence_len))

