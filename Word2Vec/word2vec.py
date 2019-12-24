import re
import nltk
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import main 
from main import *

def preprocess(paragraph):
    sentences = nltk.sent_tokenize(paragraph) # tokenize sent
    sentences = [nltk.word_tokenize(sentence) for sentence in sentences]  # tokenize each word from sentence

    for i in range(len(sentences)):     # eliminate stopwords
        sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]   
    return sentences

sentences = preprocess(dataset)
    
# size = len of vector representing each word
# window = how many words should model look at future and past for prediction
# min_count = words appearing less than 2 are to be dropped
# sg = skipgram 1 for active

# Continous Bag of Words
def CBOW(data):
    model = Word2Vec(data, min_count = 2, size = 100, window = 1) 
    model.train(data,total_examples=len(data),epochs=10)
    similar = model.similarity('learning', 'physics') # evaluating similarity 
    corr = model.wv.most_similar('machine') # in order with max to min correlation values
    print("Most similar word for research {}".format(model.wv.most_similar(positive="research", topn=1)[0][0]))
    return corr, similar

corr_CBOW, similar_CBOW = CBOW(sentences)

# Skip Gram
def SG(data):
    model = Word2Vec(data, min_count = 2, size = 100, window = 5, sg = 1) 
    model.train(data,total_examples=len(data),epochs=10)
    similar = model.similarity('learning', 'physics')
    corr = model.wv.most_similar('machine')
    print("Most similar word for research {}".format(model.wv.most_similar(positive="research", topn=1)[0][0]))
    return corr,similar

corr_SG, similar_SG = SG(sentences)
