import os
import sys
import json
import numpy as np
sys.path.append(os.getcwd())

import utils.config as config
from utils.dataset import Dictionary


def create_dictionary(qa_path):
    """ Create dictionary for question words."""
    dictionary = Dictionary()
    for path in os.listdir(qa_path):
        if 'Multiple' not in path and 'questions' in path:
            question_path = os.path.join(qa_path, path)
            qs = json.load(open(question_path))
            if not config.cp_data:
                qs = qs['questions']
            for q in qs:
                dictionary.tokenize(q['question'], True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    """ Using pre-trained glove embedding for questions. """
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('pre-trained embedding dim is {}d'.format(emb_dim))
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = map(float, vals[1:])
        word2emb[word] = np.array(list(vals))
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    d = create_dictionary(config.qa_path)
    d.dump_to_file(config.dict_path)

    d = Dictionary.load_from_file(config.dict_path)
    weights, word2emb = create_glove_embedding_init(d.idx2word, config.glove_path)
    np.save(config.glove_embed_path, weights)
