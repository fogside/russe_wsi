from utils import make_dataset, plot_attentions
from train_method import train_model
from data_generator import generate_triplet_batch
from evaluation import evaluate_weighted_ari
from sklearn.cluster import KMeans, AffinityPropagation
from gensim.models import KeyedVectors
from model import MultiCompMultiAttn
from time import gmtime, strftime
from tqdm import tqdm
import pickle
import numpy as np
import pymystem3
import pandas as pd
import os

np.random.seed(23)

if __name__ == "__main__":

    DATASET = "../data/main/bts-rnc/train.csv"
    W2V_PATH = "../models/ruscorpora_mean_hs.model.bin"
    SAVED_MODELS_PATH = "./saved_models/"
    EPOCH_NUM = 20
    CLUSTER_MODEL = 'kmeans'
    CLUSTERS_NUM = 3
    TRAIN_ON_BIG = False
    SAVE_MODELS = False
    KM_FIT_ON_BIG = False
    DRAW_PICS = False
    EVAL = True
    BINARY_EMBEDDINGS = True
    SAVE_PREDICTIONS = True
    EMB_SIZE = 300
    LEMMATIZE_TRAIN_TEXT = True
    score = None
    USE_POSTAGS = True
    EXPEREMENTS_NUM = 100
    N_COMP = 3
    INIT_WITH_W2V = False
    init_value = None
    folder_for_results_name = "k3_orthogonal_two_attens_bts-rnc_ruscorpora_emb_20epoch"
    if not os.path.exists(os.path.join('./predictions/', folder_for_results_name)):
        print("Create folder for saving results: {}".format(folder_for_results_name))
        os.mkdir(os.path.join('./predictions/', folder_for_results_name))

    mystem = pymystem3.Mystem(entire_input=False)
    cluster_models_dict = {'kmeans': KMeans(n_clusters=CLUSTERS_NUM), 'affprop': AffinityPropagation(damping=0.8)}

    df = pd.read_csv(DATASET, sep='\t')
    print("Loading embeddings....")
    wv = KeyedVectors.load_word2vec_format(W2V_PATH, binary=BINARY_EMBEDDINGS)

    true = df.gold_sense_id
    labels_true = dict()
    nets = {}

    clust_model = cluster_models_dict[CLUSTER_MODEL]

    wlist = []
    for w in df.word:
        if w not in wlist:
            wlist.append(w)

    if pd.notna(df.gold_sense_id[0]):
        for w in wlist:
            labels_true[w] = df[df.word == w].gold_sense_id.values

    if not os.path.exists(SAVED_MODELS_PATH):
        os.mkdir(SAVED_MODELS_PATH)

    print("Starting training models...")
    print("NUM of experiments = {}".format(EXPEREMENTS_NUM))
    for exp in range(1, EXPEREMENTS_NUM + 1):
        att_train = dict()
        att_test = dict()
        labels_dict = dict()
        labels_lst = list()
        print("\n\n--------------------------EXPERIMENT!{}\n\n".format(exp))
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

            if SAVE_MODELS and (not os.path.exists(os.path.join(SAVED_MODELS_PATH, "{}".format(w)))):
                os.mkdir(os.path.join(SAVED_MODELS_PATH, "{}".format(w)))

            if INIT_WITH_W2V:
                try:
                    init_value = wv[w + '_S'] if USE_POSTAGS else wv[w]
                except KeyError:
                    tmp = mystem.lemmatize(w)[0]
                    init_value = wv[tmp + '_S'] if USE_POSTAGS else wv[tmp]

                init_value = np.tile(init_value, (N_COMP, 1))
            # net = MultiCompMultiAttn(emb_size=100, n_comp=3, n_comp_mtx=4, logdir="./logdir")
            net, att_, _ = train_model(wv=wv,
                                       context_list=lines, n_comp=N_COMP, lr=1e-3, epoch_num=EPOCH_NUM,
                                       lr_decay=False, do_shuffle=False, lemmatize=LEMMATIZE_TRAIN_TEXT,
                                       word_exclude="", emb_size=EMB_SIZE,
                                       use_tfidf=False, sample_context=False, num_context_samples=-1,
                                       logdir="./logdir", restore=False,
                                       save_path=os.path.join(SAVED_MODELS_PATH, "{}/{}".format(w, w)),
                                       save_model=SAVE_MODELS, use_postags=USE_POSTAGS,
                                       net=None, init_value=init_value)
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
                                               use_postags=USE_POSTAGS,
                                               sample_context=False, num_context_samples=-1)

            for b, _ in tqdm(batch_gen, total=len(df_contexts)):
                # if len(b) == 0:
                #     b = np.random.randn(1, 100)
                p = net.predict_on_sample(b)
                att_for_word.append(p)
            att_for_word = np.array(att_for_word).squeeze()

            if KM_FIT_ON_BIG:
                clust_model.fit(np.array(att_).squeeze())
                k_pred = clust_model.predict(np.array(att_for_word))
            else:
                print(">>>>>>>>att_for_word: ", att_for_word.shape)
                k_pred = clust_model.fit_predict(np.array(att_for_word))

            labels_dict[w] = k_pred
            labels_lst.extend(k_pred)
            att_test[w] = np.array(att_for_word)

        if DRAW_PICS:
            plot_attentions(att_test, labels=labels_dict, labels_true=labels_true, save_to_path='./img/')

        if EVAL:
            score = evaluate_weighted_ari(df_file_name=DATASET, predicted_labels=labels_lst)

        if SAVE_PREDICTIONS:
            fname = "./predictions/{}/saved_wiki_{}".format(folder_for_results_name,
                                                            strftime("%Y-%m-%d_[%H-%M-%S]", gmtime()))
            if score:
                fname = fname + '_' + str(score)
            with open(fname, 'wb') as f:
                pickle.dump(labels_dict, f)
            print("Predictions has been pickled!")
