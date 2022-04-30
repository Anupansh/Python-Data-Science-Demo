# NLTK is a widely used toolkit for text and natural language processing. NLTK stands
# for Natural Language Toolkit

import nltk
from nltk.book import *
from nltk.book import FreqDist

print(text7)  # Prints the text7 corpus
print(sent7)  # Prints the first sentence of text7 corpus
print(len(sent7))
print(len(text7))
print(len(set(sent7)))
print(len(set(text7)))
print(list(set(text7))[:10])  # Display first 10 unique words from text7

# Frequency of Words
dist = FreqDist(text7)  # Counts the frequency of each word in text 7. Will return a dictionary
print(len(dist))
vocab1 = dist.keys()
print(list(vocab1)[:10])
print(dist['four']) # Will return the frequency of word four
freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100] # Will give the words with length > 5 and frequency of word more than 100
print(freqwords)

# Normalization and Stemming
# Normalization is grouping the same words as one like listen, listened, listening etc
# Stemming is finding the base words for a particular word using some algorithm

input1 = "List listed lists listing listings"
words1 = input1.lower().split(' ')
print(words1)

porter = nltk.PorterStemmer() # Example of Stemming using Porter Algorithm
print([porter.stem(t) for t in words1]) # Will return list for every word

# Lemmatization - Sometimes stemming does not produces valid words to produce base words. So, in
# order to produces meaningful base words for different words we use lemmatization

udhr = nltk.corpus.udhr.words('English-Latin1')
print(udhr[:20]) # Will give first 20 words from udhr corpus
print([porter.stem(t) for t in udhr[:20]]) # Porter Stemming will give useless words here

WNlemma = nltk.WordNetLemmatizer()
print([WNlemma.lemmatize(t) for t in udhr[:20]]) # Lemmatizer will produce the valid words after
# selecting the base words

# Tokenization using NLTK

text11 = "Children shouldn't drink a sugary drink before bed."
print(text11.split(' ')) # Standard way but not good because it will include fullstop if with last letter
# considering it as a single word

print(nltk.word_tokenize(text11)) # It will include fullstop , comma and negations differently

# Tokenizing a paragraph into sentence
text12 = "This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!"
sentences = nltk.sent_tokenize(text12)
print(len(sentences))
print(sentences)

# POS tagging - Get individual parts of speech from a sentence

# CC coordinating conjunction
# CD cardinal digit
# DT determiner
# EX existential there (like: “there is” … think of it like “there exists”)
# FW foreign word
# IN preposition/subordinating conjunction
# JJ adjective ‘big’
# JJR adjective, comparative ‘bigger’
# JJS adjective, superlative ‘biggest’
# LS list marker 1)
# MD modal could, will
# NN noun, singular ‘desk’
# NNS noun plural ‘desks’
# NNP proper noun, singular ‘Harrison’
# NNPS proper noun, plural ‘Americans’
# PDT predeterminer ‘all the kids’
# POS possessive ending parent’s
# PRP personal pronoun I, he, she
# PRP$ possessive pronoun my, his, hers
# RB adverb very, silently,
# RBR adverb, comparative better
# RBS adverb, superlative best
# RP particle give up
# TO, to go ‘to’ the store.
# UH interjection, errrrrrrrm
# VB verb, base form take
# VBD verb, past tense took
# VBG verb, gerund/present participle taking
# VBN verb, past participle taken
# VBP verb, sing. present, non-3d take
# VBZ verb, 3rd person sing. present takes
# WDT wh-determiner which
# WP wh-pronoun who, what
# WP$ possessive wh-pronoun whose
# WRB wh-abverb where, when

# Will print detail of modal auxilary
print(nltk.help.upenn_tagset('MD'))

# Getting the parts of speech
text13 = nltk.word_tokenize(text11)
print(nltk.pos_tag(text13))

text14 = nltk.word_tokenize("Visiting aunts can be a nuisance")
print(nltk.pos_tag(text14))

# Parsing sentence structure
# S stands for Sentence, NP for Noun Phrase , VP for Verb Phrase we have splitted the sentence into
# the corresponding phrases
text15 = nltk.word_tokenize("Alice loves Bob")
grammar = nltk.CFG.fromstring("""
S -> NP VP
VP -> V NP
NP -> 'Alice' | 'Bob'
V -> 'loves'
""")

parser = nltk.ChartParser(grammar)
trees = parser.parse_all(text15)
for tree in trees:
    print(tree)


from nltk.corpus import treebank
text17 = treebank.parsed_sents('wsj_0001.mrg')[0]
print(text17)


# POS tagging and parsing ambiguity
text18 = nltk.word_tokenize("The old man the boat")
nltk.pos_tag(text18)

text19 = nltk.word_tokenize("Colorless green ideas sleep furiously")
nltk.pos_tag(text19)

# An example to lemmatize a sentence with different parts of speech

# Lemmatize with POS Tag
from nltk.corpus import wordnet

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN) # Second paramter reflects optional paramter it will be returned if no key is matched


# 1. Init Lemmatizer
lemmatizer = nltk.WordNetLemmatizer()

# 2. Lemmatize Single Word with the appropriate POS tag
word = 'feet'
print(lemmatizer.lemmatize(word, get_wordnet_pos(word)))

# 3. Lemmatize a Sentence with the appropriate POS tag
sentence = "The striped bats are hanging on their feet for best"
print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])
#> ['The', 'strip', 'bat', 'be', 'hang', 'on', 'their', 'foot', 'for', 'best']

# Some important measures to calculate simmilarity

# Jaccard Distance - It is calculated as 1 - Intersection / Union between the word and the correct word. Lesser the
# distance more the simmilarity . Like in mapping and mappings unions will be 7 since p will be considered as one
# word and intersection will be 6. Thus, Jaccard distance will be 1 - 6/7 = 0.142

# Edit distance - It is the number of additions, deletions or substitutions needed to be done in order the make the
# words same for ex in lighting and drawing edit distance will be 4 since 4 subsitutuions are required

mistake = 'lighting'

print(nltk.edit_distance('drawing',mistake))
print(nltk.jaccard_distance(set('drawing'),set(mistake)))

# Demonstation of Jaccard Distance using ngrams = 3 . ngrams = 3 means word will be broken in a set of
# length of three. For ex for 'apdly' 3 gram will be ['##a','#ap','apd','pdl','dly','ly#','y##'] than on basis of these
# words distribution Jaccard distance will be calculated with the actual word i.e. apply in this case will have 3 grams
# as ['##a','#ap','app','ppl','ply','ly#','y##'] with common as 4 and total as 7 so similarity is 4 / 7 and Jaccard
# distance will be 1 - 4/7

from nltk.corpus import words
import numpy as np

correct_spellings = words.words()

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):

    correct_words = []
    for entry in entries:
        input_spell = [x for x in correct_spellings if x[0] == entry[0]]
        jaccard_dist = [nltk.jaccard_distance(set(nltk.ngrams(entry,n=3)), set(nltk.ngrams(x,n=3))) for x in input_spell]
        correct_words.append(input_spell[np.argmin(jaccard_dist)])
    return correct_words

print(answer_nine())

# Demonstration of Edit Distance

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    predicted_words = []
    for entry in entries:
        input_spell = [x for x in correct_spellings if x[0] == entry[0]]
        DL_dist = [nltk.edit_distance(x, entry) for x in input_spell]
        predicted_words.append(input_spell[np.argmin(DL_dist)])

    return predicted_words

print(answer_eleven())