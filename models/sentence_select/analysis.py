# coding: utf-8
import pandas as pd
import collections
from nltk import word_tokenize
from models.sentence_select.utils import Vocabulary
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Path to the data of in-hospital mortality task',
                    default='/data/joe/physician_notes/mimic-data/preprocessed/')
parser.add_argument('--feature_period', type=str, help='feature period', default="24",
                    choices=["24", "48", "retro"])
parser.add_argument('--feature_used', type=str, help='feature used', default="notes",
                    choices=["all", "notes", "all_but_notes"])
parser.add_argument('--note', type=str, help='feature used',
                    choices=["physician", "physician_nursing", "discharge", "all", "all_but_discharge"])
parser.add_argument('--task', type=str, help='task',
                    choices=["mortality", "readmission"])
opt = parser.parse_args()
print (opt)



#with open(f"/data/joe/physician_notes/Deep-Average-Network/{opt.note}_{opt.feature_period}_{opt.task}_vocab.pkl",'rb') as f:
#    print("----- Loading Vocab -----")
#    vocab = pickle.load(f)

a = pd.read_csv('/data/joe/physician_notes/select_sentence/logistic_regression/mortality/text_mortality_physician_24.csv')
b = a[a['y_label']==1]['bestSents'].values
text = []
for n in b:
    text.extend(word_tokenize(n.lower()))

c = collections.Counter()
text = [t for t in text if len(t) >= 2]
c.update(text)
print(c.most_common(10))
out = [i[0] for i in c.most_common(100)]
print(" ".join(out))

b = a[a['y_label']==0]['bestSents'].values
text = []
for n in b:
    text.extend(word_tokenize(n.lower()))

c = collections.Counter()
text = [t for t in text if len(t) >= 2]
c.update(text)
print(c.most_common(10))
out = [i[0] for i in c.most_common(100)]
print(" ".join(out))
