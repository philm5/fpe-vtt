"""
Precompute ngram counts of captions, to accelerate cider computation during training time.

Code from: ImageCaptioning.pytorch (https://github.com/ruotianluo/ImageCaptioning.pytorch)
"""

import os
import json
import argparse
from six.moves import cPickle
from collections import defaultdict
from official.nlp.bert import tokenization
import six

import sys
sys.path.append("cider")
from pycocoevalcap.ciderD.ciderD_scorer import CiderScorer


def get_doc_freq(refs, params):
    tmp = CiderScorer(df_mode="corpus")
    for ref in refs:
        tmp.cook_append(None, ref)
    tmp.compute_doc_freq()
    return tmp.document_frequency, len(tmp.crefs)

def pickle_dump(obj, f):
    """ Dump a pickle.
    Parameters
    ----------
    obj: pickled object
    f: file-like object
    """
    if six.PY3:
        return cPickle.dump(obj, f, protocol=2)
    else:
        return cPickle.dump(obj, f)

def build_dict(imgs, wtoi, tokenizer, params):
    wtoi['<eos>'] = 0

    count_imgs = 0

    refs_words = []
    refs_idxs = []
    for img in imgs:
        # if (params['split'] == img['split']) or \
        #     (params['split'] == 'train' and img['split'] == 'restval') or \
        #     (params['split'] == 'all'):
        #     #(params['split'] == 'val' and img['split'] == 'restval') or \
        ref_words = []
        ref_idxs = []
        for sent in img:
            # if hasattr(params, 'bpe'):
            #     sent['tokens'] = params.bpe.segment(' '.join(sent['tokens'])).strip().split(' ')
            ids = sent['caption_ids'][1:] # ignore bos token like in original code
            tmp_tokens = tokenizer.convert_ids_to_tokens(ids)
            ref_words.append(' '.join(tmp_tokens))
            ref_idxs.append(' '.join([str(id_) for id_ in ids]))
        refs_words.append(ref_words)
        refs_idxs.append(ref_idxs)
        count_imgs += 1
    print('total imgs:', count_imgs)

    ngram_words, count_refs = get_doc_freq(refs_words, params)
    ngram_idxs, count_refs = get_doc_freq(refs_idxs, params)
    print('count_refs:', count_refs)
    return ngram_words, ngram_idxs, count_refs

def main(params):

    mygts = json.load(open(params['input_json'], 'r'))
    imgs = {}
    for gt in mygts:
        vid = gt['video_id']
        if vid not in imgs:
            imgs[vid] = []

        imgs[vid].append(gt)

    imgs = list(imgs.values())

    if params['dict_type'] == 'BERT':
        tokenizer = tokenization.FullTokenizer(params['dict_file'])
        itow = tokenizer.inv_vocab
        wtoi = {w:i for i,w in itow.items()}
    elif params['dict_type'] == 'json':
        # TODO: Alter to work with my json files...
        dict_json = json.load(open(params['dict_file'], 'r'))
        itow = dict_json['ix_to_word']
        wtoi = {w:i for i,w in itow.items()}
        tokenizer = None

    ngram_words, ngram_idxs, ref_len = build_dict(imgs, wtoi, tokenizer, params)

    pickle_dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open(params['output_pkl']+'-words.p','wb'))
    pickle_dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, open(params['output_pkl']+'-idxs.p','wb'))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='Datasets/VATEX-official-split_BERT/train_samples.json', help='input json file to process into hdf5')
    parser.add_argument('--dict_type', default='BERT', help='Kind of dictionary used. E.g., json or BERT like')
    parser.add_argument('--dict_file', default='models/BERT/small_bert_bert_en_uncased_L-8_H-512_A-8_1/assets/vocab.txt', help='output json file')
    parser.add_argument('--output_pkl', default='VATEX-official-split_BERT/ngrams.pkl', help='output pickle file')
    parser.add_argument('--split', default='all', help='test, val, train, all')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict

    main(params)
