import pandas as pd
from typing import List
from sklearn.metrics import adjusted_rand_score


def gold_predict(df: pd.DataFrame):
    """ This method assigns the gold and predict fields to the data frame. """

    df = df.copy()

    df['gold'] = df['word'] + '_' + df['gold_sense_id']
    df['predict'] = df['word'] + '_' + df['predict_sense_id']

    return df


def ari_per_word_weighted(df: pd.DataFrame):
    """ This method computes the ARI score weighted by the number of sentences per word. """

    df = gold_predict(df)

    words = {word: (adjusted_rand_score(df_word.gold, df_word.predict), len(df_word))
             for word in df.word.unique()
             for df_word in (df.loc[df['word'] == word],)}

    cumsum = sum(ari * count for ari, count in words.values())
    total = sum(count for _, count in words.values())

    assert total == len(df), 'please double-check the format of your data'

    return cumsum / total, words


def evaluate_weighted_ari(df_file_name: str, predicted_labels: List):
    """ This method computes weighted ARI score for all data frame. """

    def make_result_df():
        assert len(df_file_name) != len(predicted_labels)
        df = pd.read_csv(df_file_name, sep='\t', dtype={'gold_sense_id': str, 'predict_sense_id': str})
        if type(predicted_labels[0]) is not str:
            df.predict_sense_id = [str(label) for label in predicted_labels]
        else:
            df.predict_sense_id = predicted_labels
        return df

    df = make_result_df()
    ari, words = ari_per_word_weighted(df)
    print('{}\t{}\t{}'.format('word', 'ari', 'count'))

    for word in sorted(words.keys()):
        print('{}\t{:.6f}\t{:d}'.format(word, *words[word]))

    print('\t{:.6f}\t{:d}'.format(ari, len(df)))
