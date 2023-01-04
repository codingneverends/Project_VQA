import json
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D,Flatten,Concatenate
from tensorflow.keras import Model

train_qns = open('../Dataset/Slake/train.json',encoding="utf8")
test_qns = open('../Dataset/Slake/test.json',encoding="utf8")

train_qns = json.load(train_qns)
print(len(train_qns))
test_qns = json.load(test_qns)
print(len(test_qns))

train_qns_en = [x for x in train_qns if x['q_lang'] == 'en' and x['answer_type']=='CLOSED' and x['content_type']=='Modality']
print(len(train_qns_en))
test_qns_en = [x for x in test_qns if x['q_lang'] == 'en' and x['answer_type']=='CLOSED' and x['content_type']=='Modality']
print(len(test_qns_en))

train_X = [[x["img_name"],x['question']] for x in train_qns_en]
train_Y = [x["answer"] for x in train_qns_en]
test_X = [[x["img_name"],x['question']] for x in test_qns_en]
test_Y = [x["answer"] for x in test_qns_en]

print(train_X[23],train_Y[23])

train_X=[["../Dataset/Slake/imgs/"+x[0],x[1]] for x in train_X]
test_X=[["../Dataset/Slake/imgs/"+x[0],x[1]] for x in test_X]

print(train_X[23],train_Y[23])

#PIL.Image.open(str(train_X[23][0]))
def yesno(val):
    if val.lower()=="yes":
        return 1
    return 0

train_Y=[yesno(x) for x in train_Y]
test_Y=[yesno(x) for x in test_Y]

img_height = 180
img_width = 180
train_X_qn = [x[1] for x in train_X]
test_X_qn = [x[1] for x in test_X]
train_X = [tf.keras.utils.img_to_array(tf.keras.utils.load_img(x[0],target_size=(img_width,img_height))) for x in train_X]
test_X = [tf.keras.utils.img_to_array(tf.keras.utils.load_img(x[0],target_size=(img_width,img_height))) for x in test_X]

plt.figure(figsize=(10, 10))
plt.imshow(tf.keras.utils.array_to_img(train_X[23]))

batch_size=42
train_dataset = tf.data.Dataset.from_tensor_slices((train_X,train_X_qn, train_Y)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_X,test_X_qn, test_Y)).batch(batch_size)

first_image=train_dataset.take(1)
for image,qn,feature in first_image:
    # Notice the pixel values are now in `[0,1]`.
    image=image[0]
    print(np.min(image), np.max(image))
    print(qn,feature)

num_classes = 3
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

qn_model = Sequential([
  layers.LSTM(512, dropout = 0.3,return_sequences = True,input_shape = (21,300)),
  layers.Dense(512, activation = 'relu'),
  layers.Dropout(0.3),
  layers.Dense(512, activation = 'relu'),
  layers.Dropout(0.3),
  layers.Flatten()
])

combine = Concatenate()([img_model.output,qn_model.output])
combine = Dense(128,activation='relu')(combine)
out = tf.keras.layers.Dense(2,activation='softmax')(combine)
'''
model = Sequential([ 
    layers.Concatenate()([img_model.output,qn_model.output]),
    layers.Flatten(),
    layers.Dense(2, activation='softmax')
])
'''
from keras.utils.vis_utils import plot_model
model = Model([qn_model.input,img_model.input],[out])
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

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
saved_model_path = '../models-compiled/{}_modalityclassifier_closed'.format(dataset_name.replace('/', '_'))
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