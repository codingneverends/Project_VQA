import json
import tensorflow as tf
from qn_vectors.readvec import getvec
import numpy as np

from gensim.models import KeyedVectors
wv_skipgram = KeyedVectors.load("qn_vectors/skipgram-modality-closed.model", mmap='r')

dataset_name = 'slake-test-img'
saved_model_path = '../models-compiled/{}_modalityclassifier_closed'.format(dataset_name.replace('/', '_'))
reloaded_model = tf.saved_model.load(saved_model_path)

from nltk.tokenize import word_tokenize
vec = getvec('qn_vectors/modality.vec')
def pre_process(qn):
    ar = [[0 for _j in range(100)] for _i in range(40)]
    tokens = word_tokenize(qn.lower())
    for i in range(len(tokens)):
        if tokens[i] in vec:
            for j in range(len(vec)):
                if tokens[i]==vec[j]:
                    try:
                      ar[j]=wv_skipgram.wv[vec[j]]
                    except:
                      pass
    return np.array(ar)

valid_qns = open('../Dataset/Slake/validate.json',encoding="utf8")
valid_qns = json.load(valid_qns)
valid_qns_en = [x for x in valid_qns if x['q_lang'] == 'en' and (x['answer']=='Yes' or x['answer']=='No') and x['content_type']=='Modality']
print(len(valid_qns_en))
valid_X = [[x["img_name"],pre_process(x["question"])] for x in valid_qns_en]
valid_Y = [x["answer"] for x in valid_qns_en]
valid_X=[["../Dataset/Slake/imgs/"+x[0],x[1]] for x in valid_X]

img_height = 180
img_width = 180
test_X_img = [tf.keras.utils.img_to_array(tf.keras.utils.load_img(x[0],target_size=(img_width,img_height))) for x in valid_X]
test_X_qn = [x[1] for x in valid_X]
#batch_size=42
#valid_dataset = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y)).batch(batch_size)
#print(valid_dataset.take(1))
results = reloaded_model([test_X_img,test_X_qn])
print(results)
correct=0
total=0
res_Y=[]
for res,y in zip(results,valid_Y):
    _val=0
    if res[0]>res[1]:
        _val=1
        res_Y.append("Yes")
    else:
        res_Y.append("No")
    if _val==1 and y.lower()=="yes":
        correct+=1
    if _val==0 and y.lower()=="no":
        correct+=1
    total+=1
print(valid_Y,res_Y)
print(correct/total)