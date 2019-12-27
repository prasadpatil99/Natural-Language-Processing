# Word2Vec

> *Word2vec is popular word embedding model developed by Google team in 2013 which uses neural network achitecture. <br>
Word2vec convert input words into vectors and evaluate how close each word is related to other.<br>
In short relationship between vectors are preserved by performing calculations like add, substract and distance.<br>
This relationships are preserved by iterating model through large corpus of text<br>*

Word2Vec models are shallow two layer neural networks having one input layer, one hidden layer and one output layer. <br>
Word2Vec utilizes two architectures : 

**Continous Bag of Words** - *It uses context word as input to predict the corresponding target word. The input layer contains the context words and the output layer contains the target word. <br>
The hidden layer contains the number of dimensions in which we want to represent target word present at the output layer.*<br>

![](https://www.researchgate.net/publication/324014399/figure/fig4/AS:644446416809986@1530659407943/The-CBOW-and-Skip-gram-architectures-from-15.png)

**Skip Gram** - *Uses opposite of bag of words that is it uses target word to predict context's words 
The input layer contains the target word and the output layer contains the context words.<br>
The hidden layer contains the number of dimensions in which we want to represent target word present at the input layer.*<br>

**Applications of Word2Vec :**
 - Text Generation
 - Speech Recognition
 - Question Answering
 - Sentiment Analysis
 
## Reference
https://www.pydoc.io/pypi/gensim-3.2.0/autoapi/models/word2vec/index.html
https://github.com/prasadpatil99/Web-Scrapping/blob/master/Simple-Web-Scrapping.py
## Author 
 - Prasad Patil

