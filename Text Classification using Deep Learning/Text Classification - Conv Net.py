import pandas as pd
neg_df = pd.read_excel('neg.xlsx')
pos_df = pd.read_excel('pos.xlsx')

#Merge dataset
df_ = pd.merge(neg_df,pos_df, how='outer')

#Label Encoding
from sklearn.preprocessing import LabelEncoder
lab_to_sentiment = {1:"Negative", 5:"Positive"}
def label_decoder(label):
  return lab_to_sentiment[label]
df_.Sentiment = df_.Sentiment.apply(lambda x: label_decoder(x))

#Shuffle dataset randomly
from sklearn.utils import shuffle
df = shuffle(df_)

#Remove whitespaces, digits, websites & stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
def pre(df):
    df['Text'] = df['Text'].str.replace(r'[^\w\d\s]', '')
    df['Text'] = df['Text'].str.replace(r'^https?:\/\/.*[\r\n]*', '')
    df['Text'] = [word for word in df['Text'] if not word in stopwords.words()]
    return df
data = pre(df)

#Split dataset
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size=0.7, random_state=7)

#Tokenize dataset into key value pair
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data.Text)
word_index = tokenizer.word_index

#Pad tokenized words as it has irrespective count of each other
from keras.preprocessing.sequence import pad_sequences
max_seq_length = 28
x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.Text),
                        maxlen = max_seq_length)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data.Text),
                       maxlen = max_seq_length)

#Apply label encoding to sentiments and reshaping
encoder = LabelEncoder()
encoder.fit(train_data.Sentiment.to_list())
y_train = encoder.transform(train_data.Sentiment.to_list()).reshape(-1,1)
y_test = encoder.transform(test_data.Sentiment.to_list()).reshape(-1,1)

from keras import layers
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.layers import Conv1D, Dense, Dropout

#Define sequential model
model = Sequential()

maxm_length = 55
embedding_dim = 100
#+1 due to index starting from 0
vocab_size = len(tokenizer.word_index) + 1   

#Model building 
model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxm_length))
model.add(layers.Conv1D(64, 5, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(160, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

#Compile model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fit the model
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)

#Evaluate results with test set
scores = model.evaluate(x_test, y_test, verbose=1)
