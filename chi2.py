import numpy as np
import pandas as pd
import json
import pickle
import heapq
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import norm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from nltk.tokenize import TweetTokenizer
import re
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


def clean_text(data) :
        #data.text = data.text.apply(remove_hexa_symbols)
        #data.text = data.text.apply(remove_digits)
        data = data.filter(['author', 'title', 'text']).rename(columns = {'title' : 'doc_id'})
        data["len"] = data.text.apply(lambda x: len(x))
        data.text = data.text.apply(lambda x: re.sub("All rights","",x))
        data.text = data.text.apply(lambda x: re.sub("reserved","",x))
#         data.text = data.text.apply(lambda x: re.sub("[0-9]","",x))
        data.text = data.text.apply(lambda x: re.sub("[^A-Za-z ]","",x))
        data.text = data.text.apply(lambda x: re.sub("copyright","",x))
        data.text = data.text.apply(lambda x: x.lower())
        data = data.loc[data.len > 10].reset_index()
        data.drop(columns=["len"],inplace=True)
        return data
    
def topKFrequent(nums, k):
    dic=Counter(nums)
    heapmax=[[-freq,num] for num,freq in dic.items()]
    heapq.heapify(heapmax)
    list1=[]
    for i in range(k):
        poping=heapq.heappop(heapmax)
        list1.append(poping[1])
    return list1


def get_vocab(text, max_length=400):
#     clf = CountVectorizer(lowercase=True)
#     clf.fit([text])
#     vocab = list(clf.vocabulary_.keys())
#     print("vocab before = ",vocab)
    vocab = text.split()
    k = min(max_length, len(set(vocab)))
#     return heapq.nlargest(k, vocab, key=vocab.get)
#     print(vocab)
    return topKFrequent(vocab,k)

"""
Input:  - text is a list of strings corresponding to documents
        - vocab is the vocabulary used for the problem
"""
def doc_to_dtm(text, vocab):
    #tk = TweetTokenizer()
    vectorizer = CountVectorizer(tokenizer=lambda txt: txt.split(),vocabulary=vocab) #tokenizer=tk.tokenize,
#     X = vectorizer.fit_transform(text)
    X = vectorizer.transform(text)
    return X.toarray()

def estimate_poisson(corpus):
    return np.mean(corpus,axis=0)

def accuracy(y_preds,y_true):
    return np.mean(y_preds==y_true)

def chi2(data_train,data_test):
    author1, author2 = set(data_train["author"])
    # author_1 = pd.read_csv(f'../Data/{author1}.csv').filter(['author', 'title', 'text'])
    # author_2 = pd.read_csv(f'../Data/{author2}.csv').filter(['author', 'title', 'text'])
    # data_ = pd.concat([clean_text(author_1),
    #                           clean_text(author_2)], ignore_index=True)

    # data_train = data_.sample(frac=0.7)
    # data_test = data_.drop(data_train.index)
    vocab = get_vocab(''.join([doc for doc in list(data_train["text"])]))

    text1 = data_train[data_train["author"]==author1]
    text2 = data_train[data_train["author"]==author2]

    corpus1 = doc_to_dtm(list(text1.text),vocab=vocab)
    corpus2 = doc_to_dtm(list(text2.text),vocab=vocab)

    lam_1 = estimate_poisson(corpus1)
    lam_2 = estimate_poisson(corpus2)
    y_pred = []
    for doc in list(data_test["text"]):
        dtm = doc_to_dtm([doc],vocab=vocab)
        if np.sum((dtm - lam_1)**2) < np.sum((dtm - lam_2)**2):
            y_pred.append(author1)
        else:
            y_pred.append(author2)

    y_true = list(data_test["author"])
    y_pred = [0 if item==author1 else 1 for item in y_pred]
    y_true = [0 if item==author1 else 1 for item in y_true]
    return np.array(y_pred), np.array(y_true)

# if __name__ == "__main__":
# #     author1 = "Aiyi Liu"
# #     author2 = "David Cox"
#     df = pd.DataFrame()
#     hard_pairs = []
#     while len(df) < 10:
#         author1 = random.choice(author_l)
#         author2 = random.choice(author_l)
#         if author1!=author2 and count_shared_papers(author1,author2,authors,data)==0:

#             author_1 = pd.read_csv(f'../Data/{author1}.csv').filter(['author', 'title', 'text'])
#             author_2 = pd.read_csv(f'../Data/{author2}.csv').filter(['author', 'title', 'text'])
#             n, m = author_1.shape[0], author_2.shape[0]
#             if min(n/m, m/n) >= 1/2:
#                 data_ = pd.concat([clean_text(author_1),
#                               clean_text(author_2)], ignore_index=True)

#                 data_train = data_.sample(frac=0.7)
#                 data_test = data_.drop(data_train.index)
#                 vocab = get_vocab(''.join([doc for doc in list(data_train["text"])]))

#                 text1 = data_train[data_train["author"]==author1]
#                 text2 = data_train[data_train["author"]==author2]

#                 text1
#                 #corpus1 = doc_to_dtm(["".join(list(text1.text))],vocab=vocab)
#                 corpus1 = doc_to_dtm(list(text1.text),vocab=vocab)
#                 corpus2 = doc_to_dtm(list(text2.text),vocab=vocab)

#                 lam_1 = estimate_poisson(corpus1)
#                 lam_2 = estimate_poisson(corpus2)
#                 y_pred = []
#                 for doc in list(data_test["text"]):
#                     dtm = doc_to_dtm([doc],vocab=vocab)
#                     if np.sum((dtm - lam_1)**2) < np.sum((dtm - lam_2)**2):
#                         y_pred.append(author1)
#                     else:
#                         y_pred.append(author2)


#                 """Accuracy and F1 score on test set"""
#                 y_true = list(data_test["author"])
#                 y_pred = [0 if item==author1 else 1 for item in y_pred]
#                 y_true = [0 if item==author1 else 1 for item in y_true]
#                 acc = np.mean(np.array(y_pred)==np.array(y_true))
#                 f1 = f1_score(y_pred, y_true)
#                 if acc <= 0.6 and f1 <= 0.6:
#                     hard_pairs.append((author1,author2))
#                     print(f"TESTING {author1} AGAINST {author2}")
#                     print("Accuracy on test set = ",np.mean(np.array(y_pred)==np.array(y_true)))
#                     print("f1 score = ",f1_score(y_pred, y_true))
#                     print("-----------------------------------------------------------------")
#                     df1 = pd.DataFrame({"Author 1":author1,"Author 2":author2,"Accuracy":acc,"F1":f1},index=[0])
#                     df = df.append(df1)
#                     print(df)

# with open("hard_pairs_l","rb") as f:
#     hard_pairs = pickle.load(f)

# for pair in hard_pairs:
#     author1, author2 = pair
#     y_preds, y_true = chi2(author1,author2)
#     print(f"Accuracy of {author1} against {author2} is {accuracy(y_preds,y_true)} \n")
#     print(f"F1 score = {f1_score(list(y_true),list(y_preds))}")