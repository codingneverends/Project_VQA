import json
import tensorflow as tf


dataset_name = 'slake-test-img'
saved_model_path = '../models-compiled/{}_modalityclassifier'.format(dataset_name.replace('/', '_'))
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
  
valid_qns = open('../Dataset/Slake/validate.json',encoding="utf8")
valid_qns = json.load(valid_qns)
print(len(valid_qns))
valid_qns_en = [x for x in valid_qns if x['q_lang'] == 'en']
print(len(valid_qns_en))
valid_X = [x["img_name"] for x in valid_qns_en]
valid_Y = [x["modality"] for x in valid_qns_en]
valid_X=["../Dataset/Slake/imgs/"+x for x in valid_X]
print(valid_X[0],valid_Y[0])
valid_X = [tf.keras.utils.img_to_array(tf.keras.utils.load_img(x,target_size=(180,180))) for x in valid_X]
#batch_size=42
#valid_dataset = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y)).batch(batch_size)
#print(valid_dataset.take(1))
results=reloaded_model(valid_X)
print_my_examples(valid_Y,results)