import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Embedding, LSTM, concatenate, Bidirectional
from tensorflow.keras.models import Model
import numpy as np
import json

from tensorflow.keras.preprocessing.image import ImageDataGenerator

content_types = {"Modality","Position","Organ","Size","Abnormality","Quantity","Plane","Shape","Color"}
for content_type in content_types:
    print("/n/n...... Processing - "+content_type+" ......../n/n")
    datagen = ImageDataGenerator(rotation_range=90)

    # Get data from existing dataset
    train_qns = open('../Dataset/Slake/train.json',encoding="utf8")
    train_qns = json.load(train_qns)
    train_qns = [x for x in train_qns if x['content_type']==content_type and (x['answer']=='Yes' or x['answer']=="No") and x['q_lang'] == 'en']
    if len(train_qns)==0:
        print("No data to tarin in ",content_type)
        continue
    questions = np.array([x['question'] for x in train_qns])
    image_names=["../Dataset/Slake/imgs/"+x['img_name'] for x in train_qns]
    img_width,img_height=180,180
    images = np.array([tf.keras.utils.img_to_array(tf.keras.utils.load_img(x,target_size=(img_width,img_height))) for x in image_names])
    answers = np.array([[1,0] if x['answer'] == 'Yes' else [0,1] for x in train_qns])

    # Apply augmentation to images and add them to existing dataset
    aug_images = []
    aug_answers = []
    aug_questions = []
    for i in range(len(images)):
        image = images[i]
        answer = answers[i]
        question = questions[i]
        # Apply rotation to image and add it to the dataset
        for rotated_image in datagen.flow(image.reshape(1, img_width, img_height, 3)):
            aug_images.append(rotated_image[0])
            aug_answers.append(answer)
            aug_questions.append(question)
            # If four augmented images have been added, move on to the next original image
            if len(aug_images) % 4 == 0:
                break

    # Convert lists to arrays
    aug_images = np.array(aug_images)
    aug_answers = np.array(aug_answers)

    # Concatenate original and augmented data
    images = np.concatenate((images, aug_images))
    answers = np.concatenate((answers, aug_answers))
    questions = np.concatenate((questions, aug_questions))

    #Cheking Shapes
    print("/n/nChecking Shapes/n")
    print("Images shape: ", images.shape, "Images data type: ", images.dtype)
    print("Questions shape: ", questions.shape, "Questions data type: ", questions.dtype)
    print("Answers shape: ", answers.shape, "Answers data type: ", answers.dtype)

    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Define the maximum sequence length and number of words to keep in the vocabulary
    MAX_SEQUENCE_LENGTH = 100
    MAX_NUM_WORDS = 10000

    # Create a tokenizer to convert words to integer indices
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(questions)

    # Convert the questions to sequences of integer indices
    question_sequences = tokenizer.texts_to_sequences(questions)

    # Pad the sequences to ensure they all have the same length
    padded_questions = pad_sequences(question_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    questions = padded_questions

    # Set the GloVe embedding dimension and path to the file
    EMBEDDING_DIM = 100
    GLOVE_DIR = 'glove.6B.%dd.txt' % EMBEDDING_DIM

    # Define the input layers
    image_input = Input(shape=(img_width, img_height, 3))
    text_input = Input(shape=(None,))
    num_classes=2

    # Define the image feature extraction layers
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    image_features = Dense(256, activation='relu')(x)

    # Define the textual feature extraction layers
    embedding_matrix = {}
    with open(GLOVE_DIR,encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_matrix[word] = coefs

    vocabulary_size = len(embedding_matrix.keys())
    embedding_matrix = np.zeros((vocabulary_size + 1, EMBEDDING_DIM))
    for i, vec in enumerate(embedding_matrix):
        embedding_vector = embedding_matrix[i]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    text_embedding = Embedding(vocabulary_size + 1, EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(text_input)
    text_features = Bidirectional(LSTM(256))(text_embedding)

    # Merge the image and text features
    merged_features = concatenate([image_features, text_features], axis=-1)
    merged_features = Dense(256, activation='relu')(merged_features)
    output = Dense(num_classes, activation='softmax')(merged_features)

    # Create the model
    model = Model(inputs=[image_input, text_input], outputs=output)
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    test_qns = open('../Dataset/Slake/test.json', encoding="utf8")
    test_qns = json.load(test_qns)
    test_qns = [x for x in test_qns if x['content_type']==content_type and (x['answer']=='Yes' or x['answer']=="No") and x['q_lang'] == 'en']
    test_questions = np.array([x['question'] for x in test_qns])
    test_question_sequences = tokenizer.texts_to_sequences(test_questions)
    test_questions = pad_sequences(test_question_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    test_image_names=["../Dataset/Slake/imgs/"+x['img_name'] for x in test_qns]
    test_images = np.array([tf.keras.utils.img_to_array(tf.keras.utils.load_img(x,target_size=(img_width,img_height))) for x in test_image_names])
    test_answers = np.array([[1,0] if x['answer'] == 'Yes' else [0,1] for x in test_qns])

    model.fit([images, questions], answers, epochs=10, batch_size=32,  validation_data=([test_images, test_questions], test_answers))

    #Save Model
    name = 'slake_'+content_type
    saved_model_path = '../models-compiled/{}_closed_glove'.format(name.replace('/', '_'))
    model.save(saved_model_path)
    print(content_type+" Closed Model Saved.")