import tiktoken
import nltk
from nltk.corpus import wordnet

text="Today is a great day. It is even better than yesterday. And yesterday was the best day ever."
from nltk.tokenize import sent_tokenize
a = sent_tokenize(text)
print(a)
print(nltk.word_tokenize(text))

syn = wordnet.synsets('love')
syn
syn[0].definition()
syn[0].examples()