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

unique_map = {
'Color':['Black','Dark Gray','Gray','Hyperdense','Light grey','White'],
'Modality':['CT','MRI','T1','T2','X-Ray'],
'Shape':['Circular','Irregular','Oval','U-shaped'],
'Plane':['Coronal Plane','Transverse Plane'],
'Size':['Barin','Bladder','Brain','Heart','Intestine','Large Bowel','Liver','Lung','Small Bowel'],
'Quantity':['0','1','2','3','4','5','6']
}

def get_one_hot_vector(val,arr):
    ret = []
    for i in range(len(arr)):
        if val==arr[i]:
            ret.append(1)
        else:
            ret.append(0)
    return ret

content_types = {"Modality","Size","Quantity","Plane","Shape","Color"}#"Position","Organ","Abnormality"

glob_tot=0
glob_crct=0

for content_type in content_types:
    # set up the tokenizer
    MAX_SEQUENCE_LENGTH = 100
    MAX_NUM_WORDS = 10000
    train_qns = open('../Dataset/Slake/train.json', encoding="utf8")
    train_qns = json.load(train_qns)
    train_qns = [x for x in train_qns if x['content_type']==content_type and x["answer_type"]=="OPEN" and x['q_lang'] == 'en']
    questions = np.array([x['question'] for x in train_qns])
    unique_anss = unique_map[content_type]
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(questions)
    if len(train_qns)==0:
        print("No data to valid in ",content_type)
        continue

    # Load Saved Model model
    name = 'slake_'+content_type
    saved_model_path = '../models-compiled/{}_open_glove'.format(name.replace('/', '_'))
    model = tf.keras.models.load_model(saved_model_path)

    # Load the validation data
    MAX_SEQUENCE_LENGTH = 100
    MAX_NUM_WORDS = 10000
    img_width, img_height = 180,180
    valid_qns = open('../Dataset/Slake/validate.json', encoding="utf8")
    valid_qns = json.load(valid_qns)
    valid_qns = [x for x in valid_qns if x['content_type']==content_type and x["answer_type"]=="OPEN" and x['q_lang'] == 'en']
    if len(valid_qns)==0:
        print("No data to valid in ",content_type)
        continue
    valid_questions = np.array([x['question'] for x in valid_qns])
    valid_question_sequences = tokenizer.texts_to_sequences(valid_questions)
    valid_questions = pad_sequences(valid_question_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    valid_image_names = ["../Dataset/Slake/imgs/" + x['img_name'] for x in valid_qns]
    valid_images = np.array([tf.keras.utils.img_to_array(tf.keras.utils.load_img(x, target_size=(img_width, img_height))) for x in valid_image_names])
    valid_answers = np.array([get_one_hot_vector(x["answer"],unique_anss) for x in valid_qns])

    # Predict the results
    results = model.predict([valid_images, valid_questions])
    correct = 0 
    tot = 0
    for predicted_ans,ans in zip(results, valid_answers):
        max_index = np.argmax(predicted_ans)
        crct_index = np.argmax(ans)
        if max_index == crct_index:
                correct+=1
        tot+=1
    glob_tot+=tot
    glob_crct+=correct
    print("Acurracy of "+content_type+" : ",correct/tot)
    
print("Overall Acurracy : ",glob_crct/glob_tot)
