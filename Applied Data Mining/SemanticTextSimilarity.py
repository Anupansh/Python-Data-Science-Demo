# There are two ways in which we can find similarity between the words first is lowest distance
# that is in a heirarchy tree the one with the lowest distance will have the most similarity. Heirarchy tree
# can be of noun , verb , adjective etc depending upon the type of the word
# Another method is Lowest Common Subsumer (LCS) - Lowest common subsumer is that ancestor that is closest
# to both concepts. We can use this lowest common subsumer notion to find similarity and that was proposed by
# Lin and called Lin similarity.  the formulation for doing that is if you have two concepts u and v, you
# take the log of the probability of this lowest common subsumer and divide it by some of, log of the
# probabilities of u and v. And these probabilities are something that is computed or given by the
# information content that is learnt over a large corpus.

#  If you have two words that keep appearing in very similar contexts or that could replace another word
#  in the similar context, and still the meaning remains the same, then they are more likely to be
#  semantically related

# Once you have defined the context you can compute the strength of association between words based on how
# frequently these words co-appear or how frequently they collocate. That's why it's called Collocations.
# For example, if you have two words that keep coming next to each other, then you would want to say that
# they are very highly related to each other. On the other side, if they don't occur together, then they
# are not necessarily very similar.

# Pointwise Mutual Information is defined as the log of this ratio of seeing two things together. Seeing
# the word and the context together, divided by the probability of these occurring independently. What is
# the chance that you would see the world in the overall corpus? What is the chance that you can see the
# context word in the overall corpus and what is the chance that they are actually occurring together?


import nltk
from nltk.corpus import wordnet as wn

# Using distance
#  what deer.n.01 means. It says I want deer in the sense of given by the noun meaning of it
#  and the first meaning of that.
deer = wn.synset('deer.n.01')
elk = wn.synset('elk.n.01')
horse = wn.synset('horse.n.01')
print(deer.path_similarity(elk))
print(deer.path_similarity(horse))

# Using LCS to find similarity

from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat') # You define brown_ic based on the brown_ic data. And then say, deer.lin_similarity(elk) using this brown_ic or the same way with horse with brown_ic,
print(deer.lin_similarity(elk, brown_ic))
print(deer.lin_similarity(horse, brown_ic))

# Use NLTK Collocations and Association measures
import nltk
from nltk.collocations import *
from nltk.book import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(nltk.corpus.genesis.words('english-web.txt'))
finder.apply_freq_filter(2) # finder also has other useful functions, such as frequency filter
print(finder.nbest(bigram_measures.pmi, 10)) # Top 10 pairs occuring together using the PMI measure
# from bigram_measures


