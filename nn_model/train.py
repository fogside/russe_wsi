from .model import MultiComp
from .data_generator import generate_triplet_batch
from tqdm import tqdm_notebook
import numpy as np
from typing import Iterable, Sized, Union


def train_model(context_list: Union(Iterable[str], Sized), cmp_num=3,
                lr=1e-2, epoch_num=10,
                lr_decay=False, do_shuffle=False,
                lemmatize=False, word_exclude="",
                w2v_model_path="../models/model_big_one.vec",
                use_tfidf=False, sample_context=False,
                num_context_samples=10) -> (MultiComp, np.array, np.array):

    """
    :param context_list: iterable with all contexts of main word
    :param cmp_num: number of components in att
    :param lr: learning rate
    :param epoch_num: number of epoch to train on
    :param lr_decay: use lr decay if you have more then 1 epochs
    :param do_shuffle: shuffle context_list or not
    :param lemmatize: use lemmatization of words in contexts or not
    :param word_exclude: to exclude `main_word` from train contexts
    :param w2v_model_path: path to *.vec file with trained word embeddings
    :param use_tfidf: to weight context words; fit just on all train
    :param sample_context:
    :param num_context_samples: used only if sample_context=True
    :return: trained model, attention for all samples in train, list of losses

    """

    n_comps = cmp_num
    net = MultiComp(100, n_comps)

    n_samples = len(context_list)
    train_atts = list()
    losses = []
    for epoch in range(epoch_num):

        batch_gen = generate_triplet_batch(context_list, word_exclude=word_exclude, w2v_model_path=w2v_model_path,
                                           lemmatize=lemmatize, use_tfidf=use_tfidf, do_shuffle=do_shuffle,
                                           sample_context=sample_context, num_context_samples=num_context_samples)

        pbar = tqdm_notebook(batch_gen, total=n_samples)
        print("epoch_num: ", epoch)
        for n, (sample_p, sample_n) in enumerate(pbar):
            if len(sample_p) == 0:
                continue

            loss, att = net.train_on_sample(sample_p, sample_n, lr)
            train_atts.append(att.squeeze())
            losses.append(loss)

            if n % 100 == 99:
                pbar.set_description("loss {:.3f}".format(float(np.mean(losses[-300:]))))
        if lr_decay:
            lr *= 0.707

    return net, np.array(train_atts), np.array(losses)
