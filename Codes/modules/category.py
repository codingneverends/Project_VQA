import json
import tensorflow as tf
import tensorflow_text

tf.get_logger().setLevel('ERROR')

###Reload Trained model and validating
def category(qns):
    dataset_name = 'slake-test-text'
    saved_model_path = '../../models-compiled/{}_categoryclassifier'.format(dataset_name.replace('/', '_'))
    reloaded_model = tf.saved_model.load(saved_model_path)

    content_types = {'Plane': 0, 'Size': 1, 'Abnormality': 2, 'Quantity': 3, 'Position': 4, 'Color': 5, 'Organ': 6, 'KG': 7, 'Modality': 8, 'Shape': 9}
    rev_content_types = {0: 'Plane', 1: 'Size', 2: 'Abnormality', 3: 'Quantity', 4: 'Position', 5: 'Color', 6: 'Organ', 7: 'KG', 8: 'Modality', 9: 'Shape'}

    valid_X = qns

    results=reloaded_model(valid_X)

    ret = []
    for j in range(len(results)):
        result=results[j]
        index=0
        for i in range(10):
            if result[i]>result[index]:
                index=i
        ret.append(rev_content_types[index])
    return ret