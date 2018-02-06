from xmljson import badgerfish as bf
from xml.etree.ElementTree import fromstringlist
from collections import defaultdict

class WordNetSearcher:
    def __init__(self, senses_path, syns_path):
        self.senses = WordNetSearcher._load_xml_data(senses_path)
        self.syns = WordNetSearcher._load_xml_data(syns_path)
        self.words_dict = self._make_words_dict()

    @staticmethod
    def _load_xml_data(data_path):
        with open(data_path) as f:
            data = bf.data(fromstringlist(f.readlines()))
        return data

    def _make_words_dict(self):
        synsets = defaultdict(list)

        for synset in self.syns['synsets']['synset']:
            if isinstance(synset['sense'], list):
                for sense in synset['sense']:
                    synsets[synset['@id']].append(sense['$'].lower())
            else:
                synsets[synset['@id']].append(synset['sense']['$'].lower())

            if synset['@definition'] is not None:
                synsets[synset['@id']].append(synset['@definition'])

            if (synset['@ruthes_name'] is not None) and (synset['@ruthes_name'] not in synsets[synset['@id']]):
                synsets[synset['@id']].append(synset['@ruthes_name'].lower())

        words_dict = defaultdict(dict)

        for sample in self.senses['senses']['sense']:
            if sample['@main_word'] is not None:
                words_dict[sample['@main_word'].lower()][sample['@meaning']] = synsets[sample['@synset_id']]
            else:
                words_dict[sample['@name'].lower()][sample['@meaning']] = synsets[sample['@synset_id']]

        return words_dict

    def get_synsets(self, word):
        return self.words_dict.get(word, None)



if __name__ == '__main__':

    from pprint import pprint

    nounsWordNet = WordNetSearcher(senses_path="./wordNet/senses.N.xml", syns_path="./wordNet/synsets.N.xml")
    synsets = nounsWordNet.get_synsets('банка')
    pprint(synsets.keys())
    print('\n')
    pprint(synsets)
