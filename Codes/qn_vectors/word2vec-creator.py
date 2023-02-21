from nltk.tokenize import sent_tokenize, word_tokenize
import warnings
warnings.filterwarnings(action = 'ignore')
import gensim
from gensim.models import Word2Vec
import json

# Reads ‘alice.txt’ file
sample = open("alice.txt",encoding="utf8")
s = sample.read()

train_qns = open('../../Dataset/Slake/train.json',encoding="utf8")
test_qns = open('../../Dataset/Slake/test.json',encoding="utf8")
train_qns = json.load(train_qns)
test_qns = json.load(test_qns)
train_qns_en = [x['question'] for x in train_qns if x['q_lang'] == 'en' and x['answer_type']=='CLOSED' and x['content_type']=='Modality']
print(len(train_qns_en))
test_qns_en = [x['question'] for x in test_qns if x['q_lang'] == 'en' and x['answer_type']=='CLOSED' and x['content_type']=='Modality']

# Replaces escape character with space
f = s.replace("\n", " ")

data = []

# iterate through each sentence in the file
for i in test_qns_en:
	temp = []
	print(i)
	for j in word_tokenize(i):
		temp.append(j.lower())

	data.append(temp)

# Create CBOW model
model1 = gensim.models.Word2Vec(data, min_count = 1,vector_size = 100, window = 5)

# Create Skip Gram model
model2 = gensim.models.Word2Vec(data, min_count = 1, vector_size = 100,window = 5, sg = 1)

model1.save('cbow-modality-closed.model')
model2.save('skipgram-modality-closed.model')