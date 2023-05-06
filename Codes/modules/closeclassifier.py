#Ignore Warningsss
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def CloseClassifier(img,qn,content_type):
    # set up the tokenizer
    MAX_SEQUENCE_LENGTH = 100
    MAX_NUM_WORDS = 10000
    train_qns = open('../../Dataset/Slake/train.json', encoding="utf8")
    train_qns = json.load(train_qns)
    train_qns = [x for x in train_qns if x['content_type'] == content_type and (x['answer'] == 'Yes' or x['answer'] == "No") and x['q_lang'] == 'en']
    questions = np.array([x['question'] for x in train_qns])
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(questions)

    # Load Saved Model model
    name = 'slake_'+content_type
    saved_model_path = '../../models-compiled/{}_closed_glove'.format(name.replace('/', '_'))
    model = tf.keras.models.load_model(saved_model_path)

    # Load the validation data
    MAX_SEQUENCE_LENGTH = 100
    MAX_NUM_WORDS = 10000
    img_width, img_height = 180,180
    
    questions = np.array([qn])
    question_sequences = tokenizer.texts_to_sequences(questions)
    questions = pad_sequences(question_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    img_X = [img]
    images = np.array([tf.keras.utils.img_to_array(tf.keras.utils.load_img(x, target_size=(img_width, img_height))) for x in img_X])

    results = model.predict([images, questions])
    ret = "No"
    if np.argmax(results[0])==0:
        ret="Yes"
    return ret