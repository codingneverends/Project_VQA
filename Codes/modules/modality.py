import tensorflow as tf

def modality(image):
    dataset_name = 'slake-test-img'
    saved_model_path = '../../models-compiled/{}_modalityclassifier'.format(dataset_name.replace('/', '_'))
    reloaded_model = tf.saved_model.load(saved_model_path)

    modality_types={'X-Ray': 0, 'CT': 1, 'MRI': 2}
    rev_modality_types={0: 'X-Ray', 1: 'CT', 2: 'MRI'}
    
    valid_X = image
    print(image)
    valid_X = [tf.keras.utils.img_to_array(tf.keras.utils.load_img(x,target_size=(180,180))) for x in valid_X]

    results=reloaded_model(valid_X)
    ret=[]
    for j in range(len(results)):
        result=results[j]
        index=0
        for i in range(3):
            if result[i]>result[index]:
                index=i
        ret.append(rev_modality_types[index])
    print(ret)
    return ret