import pandas as pd
import numpy as np
import json
import pickle

'''load ids of paper for each author'''
with open('../MADStat-dataset-final-version/data.json') as json_file:
    data = json.load(json_file)


'''load list of authors'''
with open('../author_name.txt') as f:
    authors = f.readlines()
authors = [author.strip() for author in authors]

'''load papers info'''
papers = pd.read_csv("../paper.csv")

def get_abstract(author_name, min_paper=30):
    author_id = authors.index(author_name)
    author_papers = data[str(author_id+1)]
    if type(author_papers) == type(1):
        return
    else:
        if len(author_papers) < min_paper:
            pass
            #print(f"{author_name} must have more than {min_paper} papers!")
        else:
            doc_id = [i - 1 for i in author_papers]
            author_df = papers.loc[doc_id,["title","sourceURL","abstract"]]
            author_df = author_df.assign(author=author_name)
            author_df.rename(columns={"abstract":"text"}, inplace=True)
            author_df.dropna(subset=['text'], inplace=True)
            if len(author_df) >= min_paper:
                author_df.to_csv(f"..//Data/{author_name}.csv")
                print(f"{author_name} has {author_df.shape[0]} papers")
                return True
            else:
                return f"{author_name} must have more than {min_paper} papers!"
              


def count_shared_papers(author1, author2,authors,data):
    author_id1 = authors.index(author1)
    author_papers1 = data[str(author_id1+1)]
    author_id2 = authors.index(author2)
    author_papers2 = data[str(author_id2+1)]
    return len(set(author_papers1).intersection(set(author_papers2)))


if __name__=="__main__":
    author_filtered = []
    for author in authors:
        output = get_abstract(author)
        if output == True:
            author_filtered.append(author)  