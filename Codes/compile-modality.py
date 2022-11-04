import json
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

train_qns = open('../Dataset/Slake/test.json',encoding="utf8")
test_qns = open('../Dataset/Slake/train.json',encoding="utf8")

train_qns = json.load(train_qns)
print(len(train_qns))
test_qns = json.load(test_qns)
print(len(test_qns))

train_qns_en = [x for x in train_qns if x['q_lang'] == 'en']
print(len(train_qns_en))
test_qns_en = [x for x in test_qns if x['q_lang'] == 'en']
print(len(test_qns_en))

train_X = [x["img_name"] for x in train_qns_en]
train_Y = [x["modality"] for x in train_qns_en]
test_X = [x["img_name"] for x in test_qns_en]
test_Y = [x["modality"] for x in test_qns_en]

print(train_X[23],train_Y[23])

train_X=["../Dataset/Slake/imgs/"+x for x in train_X]
test_X=["../Dataset/Slake/imgs/"+x for x in test_X]

print(train_X[23],train_Y[23])

PIL.Image.open(str(train_X[23]))

modality_types={'X-Ray': 0, 'CT': 1, 'MRI': 2}
rev_modality_types={0: 'X-Ray', 1: 'CT', 2: 'MRI'}
print(modality_types,rev_modality_types)

def ar_y(val):
  ar=[0,0,0]
  ar[val]=1
  return ar

train_Y=[ar_y(modality_types[x]) for x in train_Y]
test_Y=[ar_y(modality_types[x]) for x in test_Y]

print(train_X[23],train_Y[23])

img_height = 180
img_width = 180

train_X = [tf.keras.utils.img_to_array(tf.keras.utils.load_img(x,target_size=(img_width,img_height))) for x in train_X]
test_X = [tf.keras.utils.img_to_array(tf.keras.utils.load_img(x,target_size=(img_width,img_height))) for x in test_X]

plt.figure(figsize=(10, 10))
plt.imshow(tf.keras.utils.array_to_img(train_X[23]))

batch_size=42
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_Y)).batch(batch_size)

first_image=train_dataset.take(1)
for image,feature in first_image:
# Notice the pixel values are now in `[0,1]`.
  print(np.min(image), np.max(image))

num_classes = 3
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

model.summary()

epochs=10
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=epochs
)

loss, accuracy = model.evaluate(test_dataset)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

dataset_name = 'slake-test-img'
saved_model_path = '../models-compiled/{}_modalityclassifier'.format(dataset_name.replace('/', '_'))
model.save(saved_model_path, include_optimizer=False)

'''
dataset_name = 'slake-test-img'
saved_model_path = '/content/drive/My Drive/mvqa/models/{}_modalityclassifier'.format(dataset_name.replace('/', '_'))
reloaded_model = tf.saved_model.load(saved_model_path)

###Reload Trained model and validating
dataset_name = 'slake-test-img'
saved_model_path = '/content/drive/My Drive/mvqa/models/{}_modalityclassifier'.format(dataset_name.replace('/', '_'))
reloaded_model = tf.saved_model.load(saved_model_path)

modality_types={'X-Ray': 0, 'CT': 1, 'MRI': 2}
rev_modality_types={0: 'X-Ray', 1: 'CT', 2: 'MRI'}

def print_my_examples(outputs,results):
  results=tf.sigmoid(results)
  j=0
  val=0
  tot=0
  for result in results:
    output=outputs[j]
    result=results[j]
    index=0
    for i in range(3):
      if result[i]>result[index]:
        index=i
    j=j+1
    if(output==rev_modality_types[index]):
        val=val+1
    tot=tot+1
  print("Accuracy : ",val/tot)
  
valid_qns = open('tmp/Slake1.0/validate.json')
valid_qns = json.load(valid_qns)
print(len(valid_qns))
valid_qns_en = [x for x in valid_qns if x['q_lang'] == 'en']
print(len(valid_qns_en))
valid_X = [x["img_name"] for x in valid_qns_en]
valid_Y = [x["modality"] for x in valid_qns_en]
valid_X=["tmp/Slake1.0/imgs/"+x for x in valid_X]
print(valid_X[0],valid_Y[0])
valid_X = [tf.keras.utils.img_to_array(tf.keras.utils.load_img(x,target_size=(180,180))) for x in valid_X]
#batch_size=42
#valid_dataset = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y)).batch(batch_size)
#print(valid_dataset.take(1))
results=reloaded_model(valid_X)
print_my_examples(valid_Y,results)
'''