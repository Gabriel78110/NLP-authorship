import numpy as np
import pandas as pd
import json
import pickle
import time
import heapq
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import norm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from nltk.tokenize import TweetTokenizer
import re
import pickle
import random
import heapq
from get_abstract import count_shared_papers

with open('../MADStat-dataset-final-version/data.json') as json_file:
    data = json.load(json_file)
    
'''load list of authors'''
with open('../author_name.txt') as f:
    authors = f.readlines()
authors = [author.strip() for author in authors]

'''load papers info'''
papers = pd.read_csv("../paper.csv")

"""load list of authors having at least 30 papers"""
with open("../../authors","rb") as fp:
    author_l = pickle.load(fp)


def how_hard(author1,author2):
    c = 0
    a1 = pd.read_csv(f'../Data/{author1}.csv').filter(['author', 'title', 'text'])
    a2 = pd.read_csv(f'../Data/{author2}.csv').filter(['author', 'title', 'text'])
    for author in author_l:
        if (author != author1) & (author!=author2) & (min(a1.shape[0]/a2.shape[0],a2.shape[0]/a1.shape[0]) >= 1/2):
            c1 = count_shared_papers(author1,author,authors,data)
            if c1 > 0:
                c2 = count_shared_papers(author2,author,authors,data)
                if min(c1,c2) > 0:
                    c+=min(c1,c2)
    return c

def get_hardest_author(author1):
    count = 0
    cur = 0
    print(author1)
    for author_ in author_l:
        if author_ != author1:
            if count_shared_papers(author1,author_,authors,data)==0:
                if how_hard(author1,author_) > count:
                    count = how_hard(author1,author_)
                    cur = author_
                    print(count)
                    print(cur)
    return cur, count

t1 = time.time()
hard_pairs = []
while len(hard_pairs) < 10:
    author1 = random.choice(author_l)
    author2, _ = get_hardest_author(author1)
    if author2 != 0:
        hard_pairs.append((author1,author2))


with open("hard_pairs_l", "wb") as fp:
    pickle.dump(hard_pairs, fp)
t2 = time.time()
print(f"elapsed time = {t2-t1} seconds")
