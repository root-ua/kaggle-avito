from math import sqrt, exp, log
from csv import DictReader
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from pprint import pprint
from sklearn.metrics import log_loss
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from pprint import pprint
from scipy.sparse import csr_matrix

train_file = 'data/trainSearchStream.tsv'
test_file = 'data/testSearchStream.tsv'
submission = 'submission4_noshash_proba.csv'

test_count = 1

fh = FeatureHasher(n_features=20, input_type='string', non_negative=True)
v = DictVectorizer(sparse=False)
for t, line in enumerate(DictReader(open(train_file), delimiter='\t')):
    try:
        del line['ID']
    except:
        pass

    try:
        del line['IsClick']
    except:
        pass

    try:
        del line['HistCTR']
    except:
        pass

#    tr = fh.fit_transform(line)
#    pprint(tr.todense())
    tr = v.fit_transform([line])
    print(tr)

    if test_count >= 10:
        break

    test_count += 1
