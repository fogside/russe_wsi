
from .utils import make_dataset
from .train import train_model
from .data_generator import generate_triplet_batch
from sklearn.cluster import KMeans
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
import os

DATASET = "../data/main/wiki-wiki/train.csv"
W2V_PATH = "../models/model_big_one.vec"
CLUSTERS_NUM = 2
TRAIN_ON_BIG = True


df = pd.read_csv(DATASET, sep='\t')

true = df.gold_sense_id
att_train = dict()
attens = dict()
labels = []
labels2 = []
nets = {}

km = KMeans(n_clusters=CLUSTERS_NUM, random_state=23)

wlist = []
for w in df.word:
    if w not in wlist:
        wlist.append(w)

for w in wlist:

#------ TRAIN_MODEL -----------------

    if TRAIN_ON_BIG:
        path = "../data/my_data/{}_out.txt".format(w)
        if not os.path.exists(path):
            make_dataset(word=w, window=10)

        with open(path, 'r') as f:
            lines = f.readlines()
    else:
        lines = df[df.word == w].context.values

    net, att_ = train_model(context_list=lines, cmp_num=3, lr=1e-2, epoch_num=1,
                            lr_decay=False, do_shuffle=False, lemmatize=False,
                            word_exclude="", w2v_model_path=W2V_PATH,
                            use_tfidf=False, sample_context=False, num_context_samples=-1)
    att_train[w] = att_
    nets[w] = net

#------- INFER -----------------------

    pred_tmp = []
    if TRAIN_ON_BIG:
        df_contexts = df[df.word == w].context.values
    else:
        df_contexts = lines

    batch_gen = generate_triplet_batch(df_contexts, word_exclude="", w2v_model_path=W2V_PATH,
                                       lemmatize=True, use_tfidf=False, do_shuffle=False,
                                       sample_context=False, num_context_samples=-1)

    for b, _ in tqdm_notebook(batch_gen, total=len(df_contexts)):
        if len(b) == 0:
            b = np.random.randn(1, 100)
        p = net.predict_on_sample(b)
        pred_tmp.append(p)
    pred_tmp = np.array(pred_tmp).squeeze()

    km.fit(np.array(att_).squeeze())
    k_pred2 = km.predict(np.array(pred_tmp))

    k_pred = km.fit_predict(np.array(pred_tmp))

    labels2.extend(k_pred2)
    labels.extend(k_pred)
    attens[w] = np.array(pred_tmp)

