
import codecs
from collections import Counter
from cmu import CMUTweetTagger
from datetime import datetime
import fastcluster
from itertools import cycle
import json
import nltk
import numpy as np
import re
#import requests
import os
import scipy.cluster.hierarchy as sch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import metrics
#from stemming.porter2 import stem
import string
import sys
import jieba
import time
reload(sys)
sys.setdefaultencoding('utf8')


texts=["dog cat fish","dog cat cat","fish bird", 'bird']
cv = CountVectorizer()
cv_fit=cv.fit_transform(texts)
print(cv.get_feature_names())
print(cv_fit.toarray())
#['bird', 'cat', 'dog', 'fish']
#[[0 1 1 1]
# [0 2 1 0]
# [1 0 0 1]
# [1 0 0 0]]
print(cv_fit.toarray().sum(axis=0))
#[2 3 2 2]
for i in range(0,cv_fit.shape[0]):
    print cv_fit[i].sum()
print cv_fit.shape[0]
print cv_fit.shape[1]

a = Counter({'a':2,'b':3})
print len(a)










