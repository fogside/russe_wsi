from utils import make_dataset, plot_attentions
from train_method import train_model
from data_generator import generate_triplet_batch
from evaluation import evaluate_weighted_ari
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":

    DATASET = "../data/main/wiki-wiki/train.csv"
    W2V_PATH = "../models/model_big_one.vec"
    CLUSTERS_NUM = 2
    TRAIN_ON_BIG = False
    SAVE_MODELS = True
    KM_FIT_ON_BIG = False
    DRAW_PICS = True
    EVAL = True

    df = pd.read_csv(DATASET, sep='\t')
    print("Loading embeddings....")
    wv = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False)

    true = df.gold_sense_id
    att_train = dict()
    att_test = dict()
    labels_dict = dict()
    labels_lst = list()
    labels_true = dict()
    nets = {}

    km = KMeans(n_clusters=CLUSTERS_NUM, random_state=23)

    wlist = []
    for w in df.word:
        if w not in wlist:
            wlist.append(w)

    if pd.notna(df.gold_sense_id[0]):
        for w in wlist:
            labels_true[w] = df[df.word == w].gold_sense_id.values

    print("Starting training models...")
    for w in wlist:

        # ------ TRAIN_MODEL -----------------
        print("Word {}".format(w))

        if TRAIN_ON_BIG:
            path = "../data/my_data/{}_out.txt".format(w)
            if not os.path.exists(path):
                make_dataset(word=w, window=10)

            with open(path, 'r') as f:
                lines = f.readlines()
        else:
            lines = df[df.word == w].context.values

        net, att_, _ = train_model(wv=wv,
                                   context_list=lines, n_comp=3, lr=1e-2, epoch_num=1,
                                   lr_decay=False, do_shuffle=False, lemmatize=False,
                                   word_exclude="",
                                   use_tfidf=False, sample_context=False, num_context_samples=-1,
                                   logdir="./logdir", restore=False,
                                   save_path="./saved_models/{}".format(w),
                                   save_model=SAVE_MODELS)
        att_train[w] = att_
        nets[w] = net

        # ------- INFER -----------------------

        att_for_word = []

        if TRAIN_ON_BIG:
            df_contexts = df[df.word == w].context.values
        else:
            df_contexts = lines

        batch_gen = generate_triplet_batch(wv=wv,
                                           context_list=df_contexts,
                                           word_exclude="", lemmatize=True,
                                           use_tfidf=False, do_shuffle=False,
                                           sample_context=False, num_context_samples=-1)

        for b, _ in tqdm(batch_gen, total=len(df_contexts)):
            # if len(b) == 0:
            #     b = np.random.randn(1, 100)
            p = net.predict_on_sample(b)
            att_for_word.append(p)
        att_for_word = np.array(att_for_word).squeeze()

        if KM_FIT_ON_BIG:
            km.fit(np.array(att_).squeeze())
            k_pred = km.predict(np.array(att_for_word))
        else:
            k_pred = km.fit_predict(np.array(att_for_word))

        labels_dict[w] = k_pred
        labels_lst.extend(k_pred)
        att_test[w] = np.array(att_for_word)

    if DRAW_PICS:
        plot_attentions(att_test, labels=labels_dict, labels_true=labels_true, save_to_path='./img/')

    if EVAL:
        evaluate_weighted_ari(df_file_name=DATASET, predicted_labels=labels_lst)
