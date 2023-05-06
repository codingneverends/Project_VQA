#Ignore Warningsss
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_text

tf.get_logger().setLevel('ERROR')

###Reload Trained model and validating
def yesornno(qns):
    dataset_name = 'slake-yesorno'
    saved_model_path = '../../models-compiled/{}_categoryclassifier'.format(dataset_name.replace('/', '_'))
    reloaded_model = tf.saved_model.load(saved_model_path)

    answer_types = {'OPEN': 0, 'CLOSED': 1}
    rev_answer_types = {0: 'OPEN', 1: 'CLOSED'}

    valid_X = qns

    results=reloaded_model(valid_X)
    ret = []
    for j in range(len(results)):
        result=results[j]
        index=0
        for i in range(2):
            if result[i]>result[index]:
                index=i
        ret.append(rev_answer_types[index])

    return ret