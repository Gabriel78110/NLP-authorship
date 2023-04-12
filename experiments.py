import pickle
import numpy as np
from chi2 import chi2
from PHC import get_pvals
from PHC import clean_text
from PHC import accuracy
from svm_w2vec import svm
from bert import bert
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import pandas as pd


"""First load the hard pairs of authors previously computed"""
with open("easy_pairs_l","rb") as f:
    easy_pairs = pickle.load(f)

#uncomment for hard pairs
# with open("hard_pairs_l","rb") as f:
#     hard_pairs = pickle.load(f)


"""5-fold cross-validation on chi2 and PHC"""
df = pd.DataFrame()
for j, pair in enumerate(easy_pairs):
    author1, author2 = pair
    author_1 = pd.read_csv(f'../Data/{author1}.csv').filter(['author', 'title', 'text'])
    author_2 = pd.read_csv(f'../Data/{author2}.csv').filter(['author', 'title', 'text'])

    data_ = pd.concat([clean_text(author_1), clean_text(author_2)], ignore_index=True)
    kf = KFold(n_splits=5)
    acc1 = []
    acc2 = []
    acc3 = []
    acc4 = []
    for i, (train_index, test_index) in enumerate(kf.split(data_)):
        X_train = data_.iloc[train_index]
        X_test = data_.iloc[test_index]
        y_pred1, y_true1 = get_pvals(X_train,X_test)
        y_pred2, y_true2 = chi2(X_train,X_test)
        y_pred3, y_true3 = svm(X_train,X_test)
        y_pred4, y_true4 = bert(X_train,X_test)
        acc1.append(accuracy(y_true1,y_pred1))
        acc2.append(accuracy(y_true2,y_pred2))
        acc3.append(accuracy(y_true3,y_pred3))
        acc4.append(accuracy(y_true4,y_pred4))
    df1 = pd.DataFrame({"Author1":author1, "Author2":author2,"PHC acc": f"{round(np.mean(acc1),2)} ({round(np.std(acc1),2)})",\
    "Chi-square acc": f"{round(np.mean(acc2),2)} ({round(np.std(acc1),2)})", "SVM acc": f"{round(np.mean(acc3),2)} ({round(np.std(acc3),2)})",\
    "SVM-Bert acc":f"{round(np.mean(acc4),2)} ({round(np.std(acc4),2)})"},index=[j])
    df = pd.concat((df,df1))
    
    
"""Save results to csv"""
df.to_csv("Results_experiments_easy.csv")
#df.to_csv("Results_experiments_hard.csv")