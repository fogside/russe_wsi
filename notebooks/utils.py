from tqdm import tqdm_notebook
from pymystem3 import Mystem


def get_all_indexes(lst, word):
    res = []
    i = 0
    while(True):
        try:
            i = lst.index(word, i)
            res.append(i)
            i+=1
        except:
            break
    return res


def make_dataset(word, window):
    N = 1669868
    stemmer = Mystem()
    w = stemmer.lemmatize(word)[0]
    counter = 0

    with open("../data/my_data/big_one_file.txt", 'r') as bigf,\
    open("../data/my_data/{}_out.txt".format(word), 'a') as fout:
        for i in tqdm_notebook(range(N)):
            line = bigf.readline().split()
            if w in line:
                idxs = get_all_indexes(line, w)
                for i in idxs:
                    counter+=1
                    # each line is a group of neighbour words with length = 3*window
                    start = max(0, i-1-window) # if 0 is max then all before main word will be selected
                    fout.write(" ".join(line[start:i-1])+" "+" ".join(line[i:i+window])+'\n')
    return counter