import pickle
import numpy as np
from chi2 import chi2
from PHC import get_pvals
from PHC import clean_text
from PHC import accuracy
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import pandas as pd



with open("hard_pairs_l","rb") as f:
    hard_pairs = pickle.load(f)

for pair in hard_pairs:
    author1, author2 = pair
    author_1 = pd.read_csv(f'../Data/{author1}.csv').filter(['author', 'title', 'text'])
    author_2 = pd.read_csv(f'../Data/{author2}.csv').filter(['author', 'title', 'text'])

    data_ = pd.concat([clean_text(author_1), clean_text(author_2)], ignore_index=True)
    kf = KFold(n_splits=5)
    acc1 = []
    acc2 = []
    for i, (train_index, test_index) in enumerate(kf.split(data_)):
        X_train = data_.iloc[train_index]
        X_test = data_.iloc[test_index]
        y_pred1, y_true1 = get_pvals(X_train,X_test)
        y_pred2, y_true2 = chi2(X_train,X_test)
        acc1.append(accuracy(y_true1,y_pred1))
        acc2.append(accuracy(y_true2,y_pred2))
    print(f"5 folds cross-validation of our method = {np.mean(acc1)} with standard error = {np.std(acc1)}\n")
    print(f"5 folds cross-validation of chi2 method = {np.mean(acc2)} with standard error = {np.std(acc2)}")
    # data_train = data_.sample(frac=0.7)
    # data_test = data_.drop(data_train.index)

    # y_pred1, y_true1 = get_pvals(data_train,data_test)
    # y_pred2,y_true2 = chi2(data_train,data_test)
    # print(f"Accuracy of our method = {accuracy(y_true1,y_pred1)} and f1 score = {f1_score(list(y_true1),list(y_pred1))}")
    # print(f"Accuracy of chi2 method = {accuracy(y_true2,y_pred2)} and f1 score = {f1_score(list(y_true2),list(y_pred2))}")