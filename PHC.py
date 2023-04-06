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


def HC(pvals, gamma=0.25, thresh=0.5):
    pvals = np.sort(pvals[pvals <= thresh])
    N = len(pvals)
    hc = -1000
    i_star = 0
    for i in range(1,int(gamma*N)+1):
        if pvals[i-1] >= 1/N:
            num = np.sqrt(N)*((i/N) - pvals[i-1])
            den = np.sqrt((i/N)*(1-i/N))
            cur = num/den
            if cur > hc:
                hc = cur
                i_star = i
    return hc, i_star

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


def get_vocab(text, max_length=200):
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

"""Return pvals using standard normal cdf"""
def get_pvals(data_train,data_test,show_hist=False):
    author1, author2 = set(data_train["author"])
    
    def replace_labels(x):
        x[x==author1] = 1
        x[x==author2] = 0
        return x

    # author_1 = pd.read_csv(f'../Data/{author1}.csv').filter(['author', 'title', 'text'])
    # author_2 = pd.read_csv(f'../Data/{author2}.csv').filter(['author', 'title', 'text'])
    # n, m = author_1.shape[0], author_2.shape[0]
    if author1 != author2 and count_shared_papers(author1,author2,authors,data)==0:   
        # data_ = pd.concat([clean_text(author_1),
        #                                   clean_text(author_2)], ignore_index=True)

        # data_train = data_.sample(frac=0.7)
        # data_test = data_.drop(data_train.index)
        vocab = get_vocab(''.join([doc + " " for doc in list(data_train["text"])]), max_length=400)


        text1 = data_train[data_train["author"]==author1]
        text2 = data_train[data_train["author"]==author2]

        corpus1 = doc_to_dtm(list(text1.text),vocab=vocab)
        corpus2 = doc_to_dtm(list(text2.text),vocab=vocab)
        corpus_test = doc_to_dtm(list(data_test.text),vocab=vocab)
        
        lam_1 = estimate_poisson(corpus1)
        lam_2 = estimate_poisson(corpus2)
        
        sx = np.std(corpus1,axis=0)
        sy = np.std(corpus2,axis=0)
        z = (lam_1 - lam_2)/np.sqrt((sx**2/corpus1.shape[0]) + (sy**2/corpus2.shape[0]))
        z_n = (z - np.mean(z))/np.std(z)
        if show_hist:
            plt.hist(z_n)
            plt.title(f"Normalized z-counts for {author1} and {author2}")
            plt.show()
        pvals = 1 - norm.cdf(z_n)
        hc, i_star = HC(pvals)
        
        # Prediction on test set
        c1_hc = corpus1[:,pvals <= np.sort(pvals)[i_star]]
        c2_hc = corpus2[:,pvals <= np.sort(pvals)[i_star]]
        ct_hc = corpus_test[:,pvals <= np.sort(pvals)[i_star]]
        
        Z = evaluate(ct_hc,c1_hc,c2_hc)
        y_preds = predict(Z)
        y_true = replace_labels(np.array(data_test.author))   # 1 = author1, 0 = author2
        #print(f"Accuracy on test set = {accuracy(y_preds,y_true)}")
        return y_preds, y_true
    else:
        return "One author has more than twice the number of papers as the other one !!!"
        

def evaluate(new_data,c1,c2):
    return new_data - ((c1.sum(axis=0) + c2.sum(axis=0))/(c1.shape[0]+c2.shape[0]))

def predict(z):
    return np.where(z.sum(axis=1) > 0, 1, 0)

def accuracy(y_preds,y_true):
    return np.mean(y_preds==y_true)


# with open("hard_pairs_l","rb") as f:
#     hard_pairs = pickle.load(f)

# for pair in hard_pairs:
#     author1, author2 = pair
#     y_preds, y_true = get_pvals(author1,author2)
#     print(f"Accuracy of {author1} against {author2} is {accuracy(y_preds,y_true)} \n")
#     print(f"F1 score = {f1_score(list(y_true),list(y_preds))}")