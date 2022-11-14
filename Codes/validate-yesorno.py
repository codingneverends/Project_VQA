import json
import tensorflow as tf
import tensorflow_text

tf.get_logger().setLevel('ERROR')

###Reload Trained model and validating

dataset_name = 'slake-yesorno'
saved_model_path = '../models-compiled/{}_categoryclassifier'.format(dataset_name.replace('/', '_'))
reloaded_model = tf.saved_model.load(saved_model_path)

answer_types = {'OPEN': 0, 'CLOSED': 1}
rev_answer_types = {0: 'OPEN', 1: 'CLOSED'}

def print_my_result(outputs,inputs, results):
  results=tf.sigmoid(results)
  j=0
  val=0
  tot=0
  for result in results:
    result=results[j]
    output=outputs[j]
    index=0
    for i in range(2):
      if result[i]>result[index]:
        index=i
    j=j+1
    if rev_answer_types[index]==output:
      val=val+1
    tot=tot+1
  print("Accuracy : ",val/tot)

valid_qns = open('../Dataset/Slake/validate.json',encoding="utf8")
valid_qns = json.load(valid_qns)
print(len(valid_qns))
valid_qns_en = [x for x in valid_qns if x['q_lang'] == 'en']
print(len(valid_qns_en))
valid_X = [x["question"] for x in valid_qns_en]
valid_Y = [x["answer_type"] for x in valid_qns_en]
print(valid_X[0],valid_Y[0])
batch_size=42
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y)).batch(batch_size)
print(valid_dataset.take(1))

results=reloaded_model(valid_X)
print_my_result(valid_Y,valid_X, results)