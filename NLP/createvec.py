import json
from string import punctuation
import nltk
import nltk.corpus

hamlet = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')


train_qns = open('../Dataset/Slake/train.json',encoding="utf8")

train_qns = json.load(train_qns)

train_qns_en = [x['question'] for x in train_qns if x['q_lang'] == 'en' and x['content_type']=='Modality']

from nltk.tokenize import word_tokenize

train_qns_tokens = [word_tokenize(x.lower()) for x in train_qns_en]

from nltk.probability import FreqDist
fdist=FreqDist()

for sentance in train_qns_tokens:
    for word in sentance:
        fdist[word.lower()]+=1
    
print(len(fdist))

fdist_top10=fdist.most_common(40)
with open('modality.vec', 'w') as f:
    for word in fdist:
        if fdist[word]>5:
            f.write(f'{word}\n')
f.close()