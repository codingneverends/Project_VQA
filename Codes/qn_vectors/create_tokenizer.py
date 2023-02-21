import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def extract_question_features(questions, num_words=5000, max_sequence_length=100):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(questions)
    sequences = tokenizer.texts_to_sequences(questions)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequences, tokenizer

import pickle,json

# Load the questions
train_qns = open('../../Dataset/Slake/train.json',encoding="utf8")
train_qns = json.load(train_qns)

questions = [x['question'] for x in train_qns]

# Extract the features from the questions
padded_sequences, tokenizer = extract_question_features(questions)

# Save the tokenizer
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)
with open("padded_sequences.pkl", "wb") as file:
    pickle.dump(padded_sequences, file)
