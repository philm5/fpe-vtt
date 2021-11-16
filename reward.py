import tensorflow as tf
import pickle
import numpy as np
from collections import OrderedDict
from pycocoevalcap.ciderD.ciderD import CiderD
from pycocoevalcap.bleu.bleu import BleuScorer, Bleu
import json

"""
Code for calculating Cider and BLEU rewards. 
Code adapted from: https://github.com/ruotianluo/self-critical.pytorch"""

cider_reward_weight = 1.
bleu_reward_weight = 1.

CiderD_scorer = None
Bleu_scorer = None
#Cider_scorer = None

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)


def array_to_str(arr, eos_token=2):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == eos_token:
            break
    return out.strip()

def get_self_critical_reward(greedy_res, data_gts, gen_result, it=None):
    batch_size = len(data_gts)
    gen_result_size = gen_result.shape[0]
    seq_per_img = gen_result_size // len(data_gts) # gen_result_size  = batch_size * seq_per_img
    assert greedy_res.shape[0] == batch_size

    res = OrderedDict()
    #gen_result = gen_result.data.cpu().numpy()
    #greedy_res = greedy_res.data.cpu().numpy()
    for i in range(gen_result_size):
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):
        res[gen_result_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
    res__ = {i: res[i] for i in range(len(res_))}
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    gts_.update({i+gen_result_size: gts[i] for i in range(batch_size)})
    if cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts_, res_)
        print('Cider scores:', _)
    else:
        cider_scores = 0
    if bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts_, res__) #return score, scores
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0
    scores = cider_reward_weight * cider_scores + bleu_reward_weight * bleu_scores    
    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) - scores[-batch_size:][:, np.newaxis]
    scores = scores.reshape(gen_result_size)

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)
    return rewards, cider_scores