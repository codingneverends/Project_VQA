import imp
from operator import imod
import os
from string import punctuation
import nltk
import nltk.corpus

hamlet = nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')

AI='''The term artificial intelligence was coined in 1956, but AI has become more popular today thanks to increased data volumes, advanced algorithms, and improvements in computing power and storage.

Early AI research in the 1950s explored topics like problem solving and symbolic methods. In the 1960s, the US Department of Defense took interest in this type of work and began training computers to mimic basic human reasoning. For example, the Defense Advanced Research Projects Agency (DARPA) completed street mapping projects in the 1970s. And DARPA produced intelligent personal assistants in 2003, long before Siri, Alexa or Cortana were household names.

This early work paved the way for the automation and formal reasoning that we see in computers today, including decision support systems and smart search systems that can be designed to complement and augment human abilities.

While Hollywood movies and science fiction novels depict AI as human-like robots that take over the world, the current evolution of AI technologies isn’t that scary – or quite that smart. Instead, AI has evolved to provide many specific benefits in every industry. Keep reading for modern examples of artificial intelligence in health care, retail and more.'''


#Tokeization----------------------------------

from nltk.tokenize import word_tokenize

AI_tokens=word_tokenize(AI)

from nltk.probability import FreqDist
fdist=FreqDist()

for word in AI_tokens:
    fdist[word.lower()]+=1
    
#print(len(AI_tokens),len(fdist))

fdist_top10=fdist.most_common(10)
#print(fdist_top10)

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