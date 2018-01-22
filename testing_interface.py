import pandas as pd
from pymystem3 import Mystem
import gensim
import re
from typing import List
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score


def make_data(df_file_name: str, use_mystem=False,
              use_gensim_simple_preproc=True,
              throw_main_word=False, tokenize=True):
    if (use_gensim_simple_preproc and use_mystem):
        raise Exception('It is not possible to use myStem & Gensim.Preproc together!')

    if (not use_gensim_simple_preproc) and (not use_mystem):
        raise Exception('Choose at least one tokenizer to continue!')

    df = pd.read_csv(df_file_name, sep='\t', dtype={'gold_sense_id': str, 'predict_sense_id': str})
    labels = df.gold_sense_id.values
    main_words = df.word.values

    if (use_mystem):
        mystem = Mystem()
        if throw_main_word:
            sem_lem = lambda s, main_w: [w.lower() for w in mystem.lemmatize(s) if
                                         (re.match('[\w\-]+$', w) and w != main_w)]
            contexts = [sem_lem(s, w) for w, s in zip(main_words, df.context.values)]

        else:
            sem_lem = lambda s: [w.lower() for w in mystem.lemmatize(s) if re.match('[\w\-]+$', w)]
            contexts = [sem_lem(s) for s in df.context.values]

    elif (use_gensim_simple_preproc):
        contexts = [gensim.utils.simple_preprocess(s) for s in df.context.values]
        if throw_main_word:
            contexts = [[w for w in s if not re.match('^{}'.format(main_w), w)] for main_w, s in
                        zip(main_words, contexts)]

    if not tokenize:
        contexts = [' '.join(s) for s in contexts]

    return contexts, main_words, labels, list(df.word.unique)


def visualize_pca(embedded_sents: List, labels: List):
    pca = PCA()
    pca_transformed = pca.fit_transform(embedded_sents)
    plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=labels)
    plt.colorbar()
    plt.show()


def gold_predict(df):
    """ This method assigns the gold and predict fields to the data frame. """

    df = df.copy()

    df['gold']    = df['word'] + '_' + df['gold_sense_id']
    df['predict'] = df['word'] + '_' + df['predict_sense_id']

    return df

def ari_per_word_weighted(df):
    """ This method computes the ARI score weighted by the number of sentences per word. """

    df = gold_predict(df)

    words = {word: (adjusted_rand_score(df_word.gold, df_word.predict), len(df_word))
             for word in df.word.unique()
             for df_word in (df.loc[df['word'] == word],)}

    cumsum = sum(ari * count for ari, count in words.values())
    total  = sum(count for _, count in words.values())

    assert total == len(df), 'please double-check the format of your data'

    return cumsum / total, words


def evaluate_weighted_ari(df_file_name: str, predicted_labels: List):

    def make_result_df():
        assert len(df_file_name) != len(predicted_labels)
        df = pd.read_csv(df_file_name, sep='\t', dtype={'gold_sense_id': str, 'predict_sense_id': str})
        df.predict_sense_id = predicted_labels
        return df

    df = make_result_df()
    ari, words = ari_per_word_weighted(df)
    print('{}\t{}\t{}'.format('word', 'ari', 'count'))

    for word in sorted(words.keys()):
        print('{}\t{:.6f}\t{:d}'.format(word, *words[word]))

    print('\t{:.6f}\t{:d}'.format(ari, len(df)))

