import imp,json
from operator import imod
import os
from string import punctuation
import nltk
import nltk.corpus

hamlet = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')


train_qns = open('../Dataset/Slake/train.json',encoding="utf8")

train_qns = json.load(train_qns)

train_qns_en = [x['question'] for x in train_qns if x['q_lang'] == 'en' and x['content_type']=='Modality']

AI=train_qns_en[0]
#Tokeization----------------------------------

from nltk.tokenize import word_tokenize

AI_tokens=word_tokenize(AI)
print(AI_tokens)
train_qns_tokens = [word_tokenize(x) for x in train_qns_en]

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
print(fdist_top10)

from nltk.tokenize import blankline_tokenize
AI_blank=blankline_tokenize(AI)
#print(len(AI_blank),AI_blank[1])

from nltk.util import bigrams,trigrams,ngrams

string = "Intrest is the most powerful weapon in the world , if u have it then you won your goal"
quote_tokens = nltk.word_tokenize(string)
quote_ngrams=list(nltk.ngrams(quote_tokens,5))
#print(quote_tokens,quote_ngrams)

#Stemming----------------------------------
#cutting end,last of words

from nltk.stem import PorterStemmer,LancasterStemmer,SnowballStemmer
pst=PorterStemmer()
lst=LancasterStemmer()
sbst=SnowballStemmer('english')
words_to_stem = ["give","giving","given","gave"]
#for word in words_to_stem:
    #print(word+ " : " + pst.stem(word)+" : "+lst.stem(word)+" : "+sbst.stem(word))

#Lemmatization----------------------------------
#output is proper word

from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
word_lem=WordNetLemmatizer()

#print(word_lem.lemmatize('corpora'))
#for word in words_to_stem:
    #print(word+ " : " + word_lem.lemmatize(word))


#..................stopwords............

from nltk.corpus import stopwords
stopwords.words('english')

import re
punctuation = re.compile(r'[-.?!,:;()|0-9]')


post_punctuation =[]
for words in AI_tokens:
    word=punctuation.sub("",words)
    if len(word)>0:
        post_punctuation.append(word)

#print(len(post_punctuation),len(AI_tokens))

#Parts of Speech----------------------------------
#verb,Noun etc...

sentance = "John is eating a delicious cake"
sentance_tokens = word_tokenize(sentance)
setance_tags=nltk.pos_tag(sentance_tokens)
#print(setance_tags)

#NER(Named Entity Recognition)----------------------------------
#Movie,Person,Organistaion detection

from nltk import ne_chunk

ne_sentance = "The US President stays in the WHITE HOUSE"
ne_tokens=nltk.word_tokenize(ne_sentance)
ne_tags=nltk.pos_tag(ne_tokens)
ne_ner=ne_chunk(ne_tags)
#print(ne_ner)

#Syntax Tree ...................
#Chunking ...................
#grouping of words...........

new = "The big black cat ate the little mouse who was after fresh cheese"
new_tokens=nltk.pos_tag(nltk.word_tokenize(new))
grammer_np=r'NP:{<DT>?<JJ>*<NN>}'
chunk_parser=nltk.RegexpParser(grammer_np)
chunk_result=chunk_parser.parse(new_tokens)
print(chunk_result)