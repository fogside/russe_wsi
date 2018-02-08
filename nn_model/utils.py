from tqdm import tqdm
from pymystem3 import Mystem
from typing import List, Dict
import numpy as np
from matplotlib import pyplot as plt
import os
from time import gmtime, strftime
from sklearn.metrics import adjusted_rand_score

def get_all_indexes(lst: List[str], word: str) -> List[int]:
    """
    Find all occurencies of `word` in `lst`
    and return list of indexes

    :param lst: list of words
    :param word: word to find
    :return: list of indexes

    """
    res = []
    i = 0
    while (True):
        try:
            i = lst.index(word, i)
            res.append(i)
            i += 1
        except:
            break
    return res


def get_length(file_name):
    with open(file_name, 'r') as f:
        for i, v in enumerate(f):
            pass
    return i + 1


def make_dataset(word: str, window: int, big_file_path="../data/my_data/big_one_file.txt") -> int:
    """
    Create file where each line consists of `window`*2 context words
    for a particular word `word`;


    :param word: the word to create dataset for
    :param window: how much word include from each side of `word`
    :return: number of lines in new file;

    """
    stemmer = Mystem()
    w = stemmer.lemmatize(word)[0]
    counter = 0

    with open(big_file_path, 'r') as bigf, \
            open("../data/my_data/{}_out.txt".format(word), 'a') as fout:

        # needs just for tqdm;
        N = get_length(big_file_path)

        for j in tqdm(range(N)):
            line = bigf.readline().split()
            if w in line:
                idxs = get_all_indexes(line, w)
                for i in idxs:
                    counter += 1
                    # each line is a group of neighbour words with length = 2*window
                    start = max(0, i - 1 - window)  # if 0 is max then all before main word will be selected
                    fout.write(" ".join(line[start:i - 1]) + " " + " ".join(line[i:i + window]) + '\n')
    return counter


def plot_attentions(attens: Dict[str, np.array], labels: Dict[str, List[int]],
                    labels_true: Dict[str, List[int]], save_to_path: str = '.img/') -> None:
    if not os.path.exists(save_to_path):
        os.mkdir(save_to_path)

    for w, att in attens.items():
        drown = []
        for i in range(3):
            for j in range(3):
                if (i != j) and (frozenset((i, j)) not in drown):
                    drown.append(frozenset((i, j)))

                    score = adjusted_rand_score(labels_true[w], labels[w])
                    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
                    ax1.scatter(att[:, i], att[:, j], c=(labels[w] + 1) % 2, alpha=0.5, marker='o')
                    ax1.set_title('Predicted,\nscore: {}'.format(score))
                    ax2.scatter(att[:, i], att[:, j], c=labels_true[w], alpha=0.5, marker='o')
                    ax2.set_title('True')
                    plt.savefig(os.path.join(save_to_path + "{}_{}{}_{}.png".format(w, i, j, strftime("%Y-%m-%d_[%H-%M-%S]", gmtime()))))

    print("Pics have been saved to {}".format(save_to_path))
