from copy import deepcopy
from random import shuffle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from pymystem3 import Mystem
from typing import List


def generate_triplet_batch(wv: KeyedVectors,
                           context_list: List[str],
                           word_exclude="", lemmatize=False,
                           use_tfidf=False, do_shuffle=False,
                           sample_context=False, num_context_samples=10)-> (np.array, np.array):
    """
    Generate batch of positive and negative context words

    :param context_list: Iterable with full word context per sample
    :param use_tfidf: to weight context words; fit just on all train
    :param lemmatize: use lemmatization of words in contexts or not
    :param word_exclude: to exclude `main_word` from train contexts
    :param do_shuffle: shuffle context_list or not
    :param sample_context: if False use all contexts, else sample num_context_samples
    :param num_context_samples: used only if sample_context=True

    :return: yields np.arrays with shapes [any_shape, embeddings_size]
             for positive and negative samples;

    """

    stop_words = stopwords.words('russian')
    voc_size = len(wv.vocab)

    if use_tfidf:
        tfidf_tr = TfidfVectorizer()
        vals = tfidf_tr.fit_transform(context_list)

    if lemmatize:
        mystem = Mystem()

    if do_shuffle:
        context_list = deepcopy(context_list)
        shuffle(context_list)

    for j, line in enumerate(context_list):
        if lemmatize:
            line = mystem.lemmatize(line)
            line = [t for t in line if t.isalnum()]
        else:
            line = line.split()

        line = list(set(line) - set(stop_words + [word_exclude]))
        embedd_p = []

        if sample_context:
            context = np.random.choice(line, num_context_samples)
        else:
            context = line

        for i, w in enumerate(context):
            try:
                if use_tfidf:
                    num = tfidf_tr.vocabulary_[w]
                    embedd_p.append(wv[w] * vals[j, num])
                    print(w, vals[j, num], )
                else:
                    embedd_p.append(wv[w])
            except KeyError:
                continue
        embedd_n = [wv[wv.index2word[i]] for i in np.random.randint(0, voc_size - 1, len(embedd_p))]
        yield np.array(embedd_p), np.array(embedd_n)
