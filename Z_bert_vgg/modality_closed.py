import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from transformers import BertTokenizer, TFBertModel


# Getting Data to Train
import json
#'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Define an ImageDataGenerator object with rotation_range parameter
datagen = ImageDataGenerator(rotation_range=90)

# Get data from existing dataset
train_qns = open('../Dataset/Slake/train.json',encoding="utf8")
train_qns = json.load(train_qns)
train_qns = [x for x in train_qns if x['content_type']=='Modality' and (x['answer']=='Yes' or x['answer']=="No")]
questions = np.array([x['question'] for x in train_qns])
image_names=["../Dataset/Slake/imgs/"+x['img_name'] for x in train_qns]
img_width,img_height=224,224
images = np.array([tf.keras.utils.img_to_array(tf.keras.utils.load_img(x,target_size=(img_width,img_height))) for x in image_names])
answers = np.array([[1,0] if x['answer'] == 'Yes' else [0,1] for x in train_qns])

# Load VGG16 model for image feature extraction
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze all layers in VGG16
for layer in vgg16.layers:
    layer.trainable = False

# Flatten output of VGG16 for input to dense layers
flatten = Flatten()(vgg16.output)

# Add dense layers for classification
fc1 = Dense(1024, activation='relu')(flatten)
fc2 = Dense(1024, activation='relu')(fc1)
predictions = Dense(1000, activation='softmax')(fc2)

# Remove the last few layers of the VGG16 model
last_conv_layer = vgg16.get_layer('block5_conv3')
x = GlobalAveragePooling2D()(last_conv_layer.output)

# Create a new model that includes the modified layers
vgg_model = Model(vgg16.input, x)

# Load BERT tokenizer and model for question answering
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Define input layers for image and question
image_input = Input(shape=(img_width, img_height, 3))
question_input = Input(shape=(None,), dtype='string')

# Tokenize and encode question with BERT
encoded_input = tokenizer(
        K.eval(question_input).tolist(),
        padding=True,
        truncation=True,
        return_tensors='tf'
    )
question_input_tf = encoded_input['input_ids']
attention_mask_tf = encoded_input['attention_mask']
question_features = bert_model([question_input_tf, attention_mask_tf])[1]

# Extract image features with VGG16
image_features = vgg_model(image_input)

# Remove the dimensions with size 1 from the second tensor
# question_features_squeezed = tf.squeeze(question_features, axis=1)

# Concatenate the two tensors along the last dimension
concatenated_features = tf.concat([image_features, question_features], axis=-1)

# Add dense layers for answer prediction
fc3 = Dense(512, activation='relu')(concatenated_features)
fc4 = Dense(512, activation='relu')(fc3)
predictions = Dense(2, activation='softmax')(fc4)

# Create final model for visual question answering
model = Model(inputs=[image_input, question_input], outputs=predictions)

# Define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(lr=0.0001)
loss = tf.keras.losses.CategoricalCrossentropy()

# Compile model with optimizer and loss function
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.summary()

#Cheking Shapes
print("Images shape: ", images.shape, "Images data type: ", images.dtype)
print("Questions shape: ", questions.shape, "Questions data type: ",questions.dtype)
print("Answers shape: ", answers.shape, "Answers data type: ", answers.dtype)

# Train model for specified number of epochs
model.fit([images, questions], answers, epochs=100, batch_size=8)
