from model import MultiComp
from data_generator import generate_triplet_batch
from tqdm import tqdm
from gensim.models import KeyedVectors
import numpy as np
from typing import List


def train_model(wv: KeyedVectors,
                context_list: List[str], n_comp=3,
                lr=1e-2, epoch_num=10, emb_size=100,
                lr_decay=False, do_shuffle=False,
                lemmatize=False, word_exclude="",
                use_tfidf=False, sample_context=False,
                num_context_samples=10,
                logdir=None, restore=False,
                save_model=False,
                save_path="./saved_models",
                net=None
                ) -> (MultiComp, np.array, np.array):
    """
    :param wv: pretrained word2vec vectors are neesary for data generator
    :param context_list: iterable with all contexts of main word
    :param n_comp: number of components in att
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
    :param logdir: for tensorboard files; if None - nothing would be saved
    :param restore: if True then model will be restored from save_path
    :param save_model: if True model will be saved to 'save_path'
    :param save_path = "./saved_models"
    :return: trained model, attention for all samples in train, list of losses

    """
    assert (save_path is not None) if save_model else True, 'save_path must be specified if save_model is True!'

    if net is None:
        net = MultiComp(emb_size=emb_size, n_comp=n_comp, logdir=logdir, restore=restore, saved_model_path=save_path)

    n_samples = len(context_list)
    train_atts = list()
    losses = []
    for epoch in range(epoch_num):

        batch_gen = generate_triplet_batch(wv=wv, context_list=context_list,
                                           word_exclude=word_exclude,
                                           lemmatize=lemmatize, use_tfidf=use_tfidf, do_shuffle=do_shuffle,
                                           sample_context=sample_context, num_context_samples=num_context_samples)

        pbar = tqdm(batch_gen, total=n_samples)
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
    if save_model:
        net.save_model(save_path)
        print("Model has been saved!")

    return net, np.array(train_atts), np.array(losses)
