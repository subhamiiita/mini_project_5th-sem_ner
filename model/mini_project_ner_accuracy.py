#%matplotlib inline
#import matplotlib.pyplot as plt
#plt.style.use('ggplot')
#from itertools import chain

import pickle
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import nltk
from nltk.corpus import indian
from nltk.corpus.reader import ConllCorpusReader
from nltk.corpus.reader.util import *
from nltk.corpus.reader.api import * 
from nltk import chunk, tree, Tree
from nltk.tag import tnt
import os, codecs 
from nltk.internals import deprecated 
from nltk import Tree, LazyMap, LazyConcatenation 
import textwrap
class project(ConllCorpusReader):
    def _init_(self):
        super(project,self)._init_(ConllCorpusReader)
    def iob_sents(self, fileids=None, tagset=None):
        """
        :return: a list of lists of word/tag/IOB tuples
        :rtype: list(list)
        :param fileids: the list of fileids that make up this corpus
        :type fileids: None or str or list
        """
        self._require(self.WORDS, self.POS,self.NE)
        def get_iob_words(grid):
            return self._get_iob_words(grid, tagset)
        return LazyMap(get_iob_words, self._grids(fileids))
    
    def _get_iob_words(self, grid, tagset=None):
        pos_tags = self._get_column(grid, self._colmap['pos'])
        if tagset and tagset != self._tagset:
            pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
        return list(zip(self._get_column(grid, self._colmap['words']), pos_tags,self._get_column(grid,self._colmap['ne'])))
bject=ConllCorpusReader("/home/subham",'train_ner.txt',('words','pos','chunk'),('NP_B','PP','VP'))
train_sents=bject.iob_sents('train_ner.txt')
bject1=ConllCorpusReader("/home/subham",'test_accuracy.txt',('words','pos','chunk'),('NP_B','PP','VP'))
#train_sents=bject.iob_sents('conll.txt')

test_sents=bject1.iob_sents('test_accuracy.txt')
#train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
#test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
#print(test_sents[0])
#print(train_sents[0])


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = {
        'bias': 1.0,
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.istitle()': word1.istitle(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:entity': sent[i-1][2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.istitle()': word1.istitle(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:entity': sent[i+1][2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]



def sent2tokens(sent):
    return [token for token, postag,label in sent]
#print()
#print()

#print(sent2features(test_sents[0]))

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

y_token=[sent2tokens(s) for s in test_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
#picklecrf=open("miniproject.pickle","rb")
#crf=pickle.load(picklecrf)
#picklecrf.close()
crf.fit(X_train, y_train)
labels = list(crf.classes_)
labels.remove('O')
#print()
#print(labels)
#print()

y_pred = crf.predict(X_test)
#picklecrf.close()
#print(y_test)
#print()
#print()
#print()
#print("ankit");
#print(y_pred)
print()
print()
print()
file = open("output_file_predicted.txt","w",encoding='utf8')
for i,j in zip(y_token,y_pred):
    for ii,jj in zip(i,j):
        file.write(ii)
        file.write(" ")
        file.write(jj)
        file.write("\n")
file.close()
#print(X_test)
#print()
#print(y_test)
#print()
metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)

sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))

