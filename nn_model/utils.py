from tqdm import tqdm_notebook
from pymystem3 import Mystem
from typing import Iterable, List, Dict
from matplotlib import pyplot as plt
import os


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

        for j in tqdm_notebook(range(N)):
            line = bigf.readline().split()
            if w in line:
                idxs = get_all_indexes(line, w)
                for i in idxs:
                    counter += 1
                    # each line is a group of neighbour words with length = 2*window
                    start = max(0, i - 1 - window)  # if 0 is max then all before main word will be selected
                    fout.write(" ".join(line[start:i - 1]) + " " + " ".join(line[i:i + window]) + '\n')
    return counter


def plot_attentions(attens: dict, labels: Dict[List[int]] = None, save_to_path: str = '.img/'):
    if not os.path.exists(save_to_path):
        os.mkdir(save_to_path)

    for w, att in attens.items():
        if labels is None:
            labels = {w: [1] * len(att)}

        plt.scatter(att[:, 0], att[:, 2], c=labels[w], marker='o')
        plt.title("Components 0,2\nword={}".format(w))
        plt.savefig(os.path.join(save_to_path + "cmp02.png"))

        plt.scatter(att[:, 0], att[:, 1], c=labels[w], marker='o')
        plt.title("Components 0,1\nword={}".format(w))
        plt.savefig(os.path.join(save_to_path + "cmp01.png"))

        plt.scatter(att[:, 1], att[:, 2], c=labels[w], marker='o')
        plt.title("Components 1,2\nword={}".format(w))
        plt.savefig(os.path.join(save_to_path + "cmp12.png"))
