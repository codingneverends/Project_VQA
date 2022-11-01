import tensorflow_hub as hub
import tensorflow_text as text

preprocess_url='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
encoder_url='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'

text_test = ['nice movie indeed','I love pytthon programming']
bert_preprocess_model=hub.KerasLayer(preprocess_url)
text_preproccessed=bert_preprocess_model(text_test)
bert_model=hub.KerasLayer(encoder_url)
bert_results=bert_model(text_preproccessed)
print(bert_results.keys())