import json
import tensorflow as tf
import tensorflow_text

tf.get_logger().setLevel('ERROR')

###Reload Trained model and validating

dataset_name = 'slake-test-text'
saved_model_path = '../models-compiled/{}_categoryclassifier'.format(dataset_name.replace('/', '_'))
reloaded_model = tf.saved_model.load(saved_model_path)

content_types = {'Plane': 0, 'Size': 1, 'Abnormality': 2, 'Quantity': 3, 'Position': 4, 'Color': 5, 'Organ': 6, 'KG': 7, 'Modality': 8, 'Shape': 9}
rev_content_types = {0: 'Plane', 1: 'Size', 2: 'Abnormality', 3: 'Quantity', 4: 'Position', 5: 'Color', 6: 'Organ', 7: 'KG', 8: 'Modality', 9: 'Shape'}

def print_my_result(outputs,inputs, results):
  qns=[0,0,0,0,0,0,0,0,0,0]
  crcts = [0,0,0,0,0,0,0,0,0,0]
  results=tf.sigmoid(results)
  j=0
  val=0
  tot=0
  for result in results:
    result=results[j]
    output=outputs[j]
    qns[content_types[output]]+=1
    index=0
    for i in range(10):
      if result[i]>result[index]:
        index=i
    j=j+1
    if rev_content_types[index]==output:
      crcts[content_types[output]]+=1
      val=val+1
    else:
      print(str(result)+" : "+output+" : "+inputs[j])
    tot=tot+1
  _val = [x/y for x,y in zip(crcts,qns)]
  for i in range(10):
    print(rev_content_types[i]+" : "+str(_val[i]*100)+"%")
  print("Accuracy : ",val/tot)

valid_qns = open('../Dataset/Slake/train.json',encoding="utf8")
valid_qns = json.load(valid_qns)
print(len(valid_qns))
valid_qns_en = [x for x in valid_qns if x['q_lang'] == 'en']
print(len(valid_qns_en))
valid_X = [x["question"] for x in valid_qns_en]
valid_Y = [x["content_type"] for x in valid_qns_en]
print(valid_X[0],valid_Y[0])
batch_size=42
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y)).batch(batch_size)
print(valid_dataset.take(1))

results=reloaded_model(valid_X)
print_my_result(valid_Y,valid_X, results)