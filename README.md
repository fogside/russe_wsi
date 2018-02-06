# RUSSE 2018 Word Sense Induction and Disambiguation Shared Task

This repository contains models and notebooks created during participation in the [shared task on word sense induction and disambiguation for the Russian language](http://russe.nlpub.org/2018/wsi). **TLDR**: You are given a word, e.g. ```"замок"``` and a bunch of text fragments (aka "contexts") where this word occurrs, e.g. ```"замок владимира мономаха в любече"``` and  ```"передвижению засова ключом в замке"```. You need to cluster these contexts in the (unknown in advance) number of clusters which correspond to various senses of the word. In this example you want to have two groups with the contexts of the "lock" and the "castle" senses of the word ```"замок"```. 


Description of the datasets
--------

The participants of the shared task need to work with three datasets of varying sense inventories and types of texts. All the datasets are located in the directory ```data/main```. One dataset is located in one directory. The name of the directory is ```<inventory>-<corpus>```. For instance ```bts-rnc```, which represents datasets based on the word sense inventory BTS (Bolshoi Tolkovii Slovar') and the RNC corpus. Here is the list of the datasets:

1. **wiki-wiki** located in ```data/main/wiki-wiki```: This dataset contains contexts from Wikipedia articles. The senses of this dataset correspond to a subset of Wikipedia articles.

2. **bts-rnc** located in ```data/main/bts-rcn```:
This dataset contains contexts from the Russian National Corpus (RNC). The senses of this dataset correspond to the senses of the Gramota.ru online dictionary (and are equivalent to the senses of the Bolshoi Tolkovii Slovar, BTS).

3. **active-dict** located in ```data/main/active-dict```: The senses of this dataset correspond to the senses of the Active Dictionary of the Russian Language a.k.a. the 'Dictionary of Apresyan'. Contexts are extracted from examples and illustrations sections from the same dictionary.


For the three datasets described above, we will release test parts which will be used for the computation of the final results and for ranking the participants. Note that **in the test part, we will provide new words: the train datasets do not contain examples of the words in the test datasets**.

In addition, in the directory ```data/additional```, we provide three extra datasets, which can be used as additional training data from (Lopukhin and Lopukhina, 2016). These datasets are based on various sense inventories (active dictionary, BTS) and various corpora (RNC, RuTenTen). Note that we will not release any test datasets that correspond to these datasets (yet they still may be useful e.g. for multi-task learning).  

The table below summarizes the datasets:

|Dataset|Type|Inventory|Corpus|Split|Num. of words|Num. of senses|Avg. num. of senses|Num. of contexts|
|-----|-----|---------|-----|------|:---------:|:----------:|:----------:|:----------:|
|wiki-wiki|main|Wikipedia|Wikipedia|train|4|8|2.0|439
|bts-rnc|main|Gramota.ru|RNC|train|30|96|3.2|3491
|active-dict|main|Active Dict.|Active Dict.|train|85|312|3.7|2073
|active-rnc|additional|Active Dict.|RNC|train|20|71|3.6|1829
|active-rutenten|additional|Active Dict.|ruTenTen|train|21|71|3.4|3671
|bts-rutenten|additional|Gramota.ru|ruTenTen|train|11|25|2.3|956

Format of the dataset files
----------

Train and test datasets are stored in .csv files (the name of the folder corresponds to the name of the dataset), each file has a header:

```
context_id    word    gold_sense_id    predict_sense_id    positions    context
```

**Type of the file and dialect**: CSV (TSV): tab separated,  with a header, no quote chars for fields, one row is one line in the file (no multi-line rows are supported).

**Encoding**: utf-8

**Target**: ```predict_sense_id``` is the prediction of a system (this field is not filled by default)

**Sample**:

```
context_id    word    gold_sense_id    predict_sense_id    positions    context

1    граф    2        0-3,132-137    Граф -- это структура данных. Кроме этого, в дискретной математике теория графов ...

...    ...    ...    ...    ...    ...
```

Structure of this repository
---------------------------

- ```data``` -- directory with the train datasets and the corresponding baselines based on the Adagram
- ```notebooks``` -- directory with all experiments
- ```nn_model``` -- python files of tensorflow model
- ```organizers_baseline``` -- folder with organizers baseline scripts
- ```trash``` -- several scripts which are not used now