'''
Created on Oct 26, 2015

@author: jcheung

Developed for Python 2. May work for Python 3 too (but I never tried) with minor changes.
'''
import xml.etree.cElementTree as ET
import codecs
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus.reader.wordnet import WordNetError
import difflib


class WSDInstance:
    def __init__(self, my_id, lemma, context, index, pos=None):
        self.id = my_id  # id of the WSD instance
        self.lemma = lemma  # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.pos = pos
        self.index = index  # index of lemma within the context

    def __str__(self):
        '''
        For printing purposes.
        '''
        return "{}\t{} ({})\t{}\t{}".format(self.id, self.lemma, self.pos,' '.join(self.context), self.index)


def load_instances(f):
    '''
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    '''
    tree = ET.parse(f)
    root = tree.getroot()

    dev_instances = {}
    test_instances = {}

    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            context = [el.attrib['lemma'] for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = el.attrib['lemma']
                    pos = el.attrib['pos']
                    instances[my_id] = WSDInstance(my_id, lemma, context, i, pos)
    return dev_instances, test_instances


def load_key(f):
    '''
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys. 
    '''
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1: continue
        # print (line)
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key


# Not used
def to_ascii(s):
    # remove all non-ascii characters
    return codecs.encode(s, 'ascii', 'ignore')


def most_frequent(instances):
    predict = {}
    for key, inst in instances.items():
        predict[key] = wn.synsets(inst.lemma)[0]

    return predict


def lesks(instances, fn, pos=False):
    predict = {}
    for key, inst in instances.items():
        if pos: predict[key] = fn(inst.context, inst.lemma, pos=inst.pos)
        else: predict[key] = fn(inst.context, inst.lemma)

    return predict


def accuracy(y, y_pred):
    n = len(y)
    k = 0
    for key in y.keys():
        try:
            senses = [wn.synset_from_sense_key(sense) for sense in y[key]]
            if y_pred[key] in senses:
                k += 1
            #else: print(y_pred[key], senses)   # Why does wordnet.synset_from_sense_key return 'brazil_nut.n.02'?
            # Are these labels all right?
        except WordNetError:
            pass # This should only occur for "budget"

    return k/n


def preprocess_instances(instances):
    # In-place
    for key, inst in instances.items():
        inst.lemma = inst.lemma.lower()
        post = []
        for token in inst.context:
            if len(token) > 1 and token not in stops:
                post.append(token.lower())
        inst.context = post


'''
'''
def modified_lesk(context_sentence, ambiguous_word, pos=None, synsets=None):
    '''
    Modified from NLTK's implementation of Lesk's algorithm by Liling Tan and Dmitrijs Milajevs.

    '''
    context = set(context_sentence)

    if synsets is None:
        synsets = wn.synsets(ambiguous_word)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None
    try:
        s = []
        for ss in synsets:
            count = len(context.intersection(ss.definition().split()))

            for ssdef in ss.definition().split():
                ssdefsyn = wn.synsets(ssdef)
                if ssdefsyn:
                    ssdefsyn = ssdefsyn[0]
                    count += len(context.intersection(ssdefsyn.definition().split()))

                    for condef in context:
                        condefsyn = wn.synsets(condef)
                        if condefsyn:
                            condefsyn = condefsyn[0]
                            # count += len(set(condefsyn.definition().split()).intersection(ss.definition().split()))
                            count += len(set(ssdefsyn.definition().split()).intersection(condefsyn.definition().split()))
            s.append((count, ss))
        _, sense = max(s)

        # _, sense = max(
        #     (len(context.intersection(ss.definition().split().extend(wn.synsets(ssdef)[0].definition().split() if wn.synsets(ssdef) else []))), ss)
        #     for ss in synsets
        #     for ssdef in (ss.definition().split() if ss.definition() else [])
        # )

    except TypeError as e:
        raise e

    return sense


def get_closest_lemma(word):
    return difflib.get_close_matches(word, wn.all_lemma_names())[0]


if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for k, v in dev_instances.items() if k in dev_key}
    test_instances = {k: v for k, v in test_instances.items() if k in test_key}

    # My code:
    stops = set(stopwords.words('english'))

    # print(stops)

    preprocess_instances(dev_instances)
    preprocess_instances(test_instances)
    # txt = ""
    # for v in dev_instances.values():
    #     txt += v.lemma + " "
    # print(len(test_instances))

    baseline_pred_dev = most_frequent(dev_instances)
    baseline_pred_test = most_frequent(test_instances)
    print("Baseline acc. dev: {}\t test: {}".format(accuracy(dev_key, baseline_pred_dev), accuracy(test_key, baseline_pred_test)))

    lesk_pred_dev = lesks(dev_instances, lesk)
    lesk_pred_test = lesks(test_instances, lesk)
    print("Lesk acc. dev: {}\t test: {}".format(accuracy(dev_key, lesk_pred_dev), accuracy(test_key, lesk_pred_test)))

    # All POS are varieties of noun!
    # leskpos_pred_dev = lesks(dev_instances, lesk, pos=True)
    # leskpos_pred_test = lesks(test_instances, lesk, pos=True)
    # print("Lesk w/POS acc. dev: {}\t test: {}".format(accuracy(dev_key, leskpos_pred_dev), accuracy(test_key, leskpos_pred_test)))

    depth_lesk_pred_dev = lesks(dev_instances, modified_lesk)
    depth_lesk_pred_test = lesks(test_instances, modified_lesk)
    print("Modified Lesk acc. dev: {}\t test: {}".format(accuracy(dev_key, depth_lesk_pred_dev), accuracy(test_key, depth_lesk_pred_test)))

    # depth_leskpos_pred_dev = lesks(dev_instances, modified_lesk, pos=True)
    # depth_leskpos_pred_test = lesks(test_instances, modified_lesk, pos=True)
    # print("Modified Lesk w/POS acc. dev: {}\t test: {}".format(accuracy(dev_key, depth_lesk_pred_dev), accuracy(test_key, depth_lesk_pred_test)))
