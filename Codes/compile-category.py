import zipfile,json
import os
import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

train_qns = open('../Dataset/Slake/train.json',encoding="utf8")
test_qns = open('../Dataset/Slake/test.json',encoding="utf8")

train_qns = json.load(train_qns)
print(len(train_qns))
test_qns = json.load(test_qns)
print(len(test_qns))

train_qns_en = [x for x in train_qns if x['q_lang'] == 'en']
print(len(train_qns_en))
test_qns_en = [x for x in test_qns if x['q_lang'] == 'en']
print(len(test_qns_en))

train_X = [x["question"] for x in train_qns_en]
train_Y = [x["content_type"] for x in train_qns_en]
test_X = [x["question"] for x in test_qns_en]
test_Y = [x["content_type"] for x in test_qns_en]

print(train_X[23],train_Y[23])

content_types = {'Plane': 0, 'Size': 1, 'Abnormality': 2, 'Quantity': 3, 'Position': 4, 'Color': 5, 'Organ': 6, 'KG': 7, 'Modality': 8, 'Shape': 9}
rev_content_types = {0: 'Plane', 1: 'Size', 2: 'Abnormality', 3: 'Quantity', 4: 'Position', 5: 'Color', 6: 'Organ', 7: 'KG', 8: 'Modality', 9: 'Shape'}
print(content_types)
print(rev_content_types)

def ar_y(val):
  ar=[0,0,0,0,0,0,0,0,0,0]
  ar[val]=1
  return ar

train_Y=[ar_y(content_types[x]) for x in train_Y]
test_Y=[ar_y(content_types[x]) for x in test_Y]

batch_size=42
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).batch(batch_size)

print(train_dataset.take(1))

for text_batch, label_batch in train_dataset.take(1):
  for i in range(3):
    print(f'Review: {text_batch.numpy()[i]}')
    label = label_batch.numpy()[i]
    j=0
    for j in range(10):
      if label[j]==1:
        print(f'Label : {label} ({rev_content_types[j]})')

tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

text_test = ['What color does the left lung show in the picture?']
text_preprocessed = bert_preprocess_model(text_test)

print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

bert_model = hub.KerasLayer(tfhub_handle_encoder)

bert_results = bert_model(text_preprocessed)

print(f'Loaded BERT: {tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')

def build_classifier_model(num_classes):
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(num_classes, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

num_classes=10
classifier_model = build_classifier_model(num_classes)
bert_raw_result = classifier_model(tf.constant(text_test))
print(tf.sigmoid(bert_raw_result))

#tf.keras.utils.plot_model(classifier_model)

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

epochs = 5
steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')
classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

print(f'Training model with {tfhub_handle_encoder}')
history = classifier_model.fit(x=train_dataset,
                               validation_data=test_dataset,
                               epochs=epochs)

loss, accuracy = classifier_model.evaluate(test_dataset)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

history_dict = history.history
print(history_dict.keys())

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(2, 1, 1)
# r is for "solid red line"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
# plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

dataset_name = 'slake-test-text'
saved_model_path = '../models-compiled/{}_categoryclassifier'.format(dataset_name.replace('/', '_'))

classifier_model.save(saved_model_path, include_optimizer=False)

reloaded_model = tf.saved_model.load(saved_model_path)

def print_my_examples(inputs, results):
  results=tf.sigmoid(results)
  j=0
  for result in results:
    input=inputs[j]
    result=results[j]
    index=0
    for i in range(10):
      if result[i]>result[index]:
        index=i
    j=j+1
    print(input,rev_content_types[index])


examples = [
    'What is the scanning plane of this image?',
    'Does the picture contain liver?',
    'Which part of the body does this image belong to?',
    'What modality is used to take this image?',
    'What is the shape of the kidney in the picture?',
    'How many kidneys in this image?',
    'What is the largest organ in the picture?',
    'What color is the lung in the picture?',
    'Is the lung healthy?',
    'What is the effect of the main organ in this picture?',
  'What is the main organ in the image?',
  'Does the picture contain liver?',
  'What diseases are included in the picture?',
  'Where is/are the abnormality located?',
  'Which is the biggest in this image,lung,liver or heart?',
  'What modality is used to take this image?',
  'What color does the right lung show in the picture?',
  'What color does the left lung show in the picture?'
]

reloaded_results = reloaded_model(tf.constant(examples))

print('Results from the saved model:')

print_my_examples(examples, reloaded_results)