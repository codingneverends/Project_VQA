import tensorflow as tf
import numpy as np
from extract_question_feature import extract_single_question_feature
import pickle
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import json
train_qns = open('../../Dataset/Slake/train.json',encoding="utf8")
train_qns = json.load(train_qns)
train_qns = [x for x in train_qns if x['content_type']=='Modality' and (x['answer']=='Yes' or x['answer']=="No")]

# Load the questions
questions = [x['question'] for x in train_qns]
# Extract the features from the questions

# Load the tokenizer
with open('tokenizer.pkl', "rb") as file:
    tokenizer = pickle.load(file)

# Load the padded sequences
with open('padded_sequences.pkl', "rb") as file:
    padded_sequences = pickle.load(file)

# Encode the question
sequence = tokenizer.texts_to_sequences(questions)

# Pad the encoded question
padded_sequences = pad_sequences(sequence, maxlen=100)

# Load the image features
train_X=["../../Dataset/Slake/imgs/"+x['img_name'] for x in train_qns]
img_height = 180
img_width = 180
train_img = [tf.keras.utils.img_to_array(tf.keras.utils.load_img(x,target_size=(img_width,img_height))) for x in train_X]
image_data = np.array([x for x in train_img])

# Build the model
question_input = tf.keras.layers.Input(shape=(100,))

img_model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten()
])

padded_sequences = tf.cast(padded_sequences, dtype = tf.float32)

merged_inputs = tf.keras.layers.concatenate([img_model.output, question_input])
expanded_input = tf.expand_dims(merged_inputs, axis=-1)
x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64))(expanded_input)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
output = tf.keras.layers.Dense(2, activation="softmax")(x)

model = tf.keras.Model(inputs=[img_model.input, question_input], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

def yesno(val):
    if val.lower()=="yes":
        return [1,0]
    return [0,1]

# Train the model
labels = np.array([yesno(x['answer']) for x in train_qns])
model.fit([image_data, padded_sequences], labels, epochs=10, batch_size=8)
model.save('glove-img', include_optimizer=False)
