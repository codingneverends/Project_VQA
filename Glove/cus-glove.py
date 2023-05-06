import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from glove import Glove
from glove import Corpus

# Load sample questions
import json
train_qns = open('../Dataset/Slake/train.json',encoding="utf8")
train_qns = json.load(train_qns)
train_qns = [x for x in train_qns if x['content_type']=='Modality' and (x['answer']=='Yes' or x['answer']=="No")]
questions = np.array([x['question'] for x in train_qns])

# Tokenize questions
tokenized_questions = [question.split() for question in questions['text']]

# Train Word2Vec model on tokenized questions
model = Word2Vec(tokenized_questions, size=100, window=5, min_count=1, workers=4)

# Build corpus for GloVe model
corpus = Corpus()
corpus.fit(tokenized_questions, window=5)

# Train GloVe model on corpus
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=50, no_threads=4, verbose=True)
glove.add_dictionary(model.wv.vocab)

# Save trained model
glove.save('custom_glove.model')
