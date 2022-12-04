import os
from os.path import join
import json
import numpy as np
from src.utils.configs import cfg
import pandas as pd

root_dir, _ = os.path.split(os.path.abspath(__file__))
root_dir = os.path.dirname(root_dir)
root_dir = os.path.dirname(root_dir)
root_dir = os.path.dirname(root_dir)

tesa_dict_file = join(root_dir, 'dataset/processed/mimic3_dict.json')
with open(tesa_dict_file, 'r', encoding='utf-8') as f:
    tesa_dict = json.load(f)

glove_vectors_file = join(root_dir, 'dataset/baselines/GloVe/mimic3/mimic3_vectors_6.txt')

med2vec_vectors_file = join(root_dir, 'dataset/baselines/med2vec/mimic3/outputs_med2vec.9.npz')
med2vec_origin_dict_file = join(root_dir, 'dataset/baselines/med2vec/mimic3/outputs.types')

mce_vectors_file = join(root_dir, 'dataset/baselines/MCE/mimic3/mimic3-attn1-e10-ne10-w6-aw20_month.vec')
mce_origin_dict_file = join(root_dir, 'dataset/baselines/MCE/mimic3/mmimic3_vocabs.csv')

cbow_file = join(root_dir, 'dataset/baselines/CBOW/mimic3/cbow_sk_6_epoch_10.vect')
sg_file = join(root_dir, 'dataset/baselines/SG/mimic3/sg_sk_6_epoch_10.vect')
tesan_file = join(root_dir, 'outputs/tasks/embedding/tesan/vects/mimic3_model_tesan_epoch_1_sk_6.vect')
sa_file = join(root_dir, 'outputs/tasks/embedding/sa/vects/mimic3_model_sa_epoch_30_sk_6.vect')
normal_file = join(root_dir, 'outputs/tasks/embedding/normal/vects/mimic3_model_normal_epoch_30_sk_6.vect')
interval_file = join(root_dir, 'outputs/tasks/embedding/interval/vects/mimic3_model_interval_epoch_30_sk_6.vect')


# TODO Generate vector embeddings from source code for all baseline models
def glove_trans():
    coed_weights = dict()
    with open(glove_vectors_file, 'r') as read_file:
        for line in read_file:
            line2list = line.split()
            code = line2list[0]
            weight = line2list[1:]
            coed_weights[code] = weight
    weights = []
    for k, v in tesa_dict.items():
        if v not in coed_weights.keys():
            weights.append(coed_weights['<unk>'])
        else:
            weights.append(coed_weights[v])
    weights = np.array(weights, dtype=float)
    print(weights.shape)
    return weights


def med2vec_trans():
    origin_weights = np.load(med2vec_vectors_file)
    origin_weights = origin_weights['W_emb']
    embedding_size = origin_weights.shape[1]
    padding_array = np.zeros(embedding_size)
    with open(med2vec_origin_dict_file) as read_file:
        origin_dict = json.load(read_file)
    new_dict = {}
    for k in origin_dict.keys():
        key = k.replace('.', '')
        new_dict[key] = origin_dict[k]
    weights = []
    padding_count = 0
    for k, v in tesa_dict.items():
        if v not in new_dict.keys():
            weights.append(padding_array)
            padding_count += 1
        else:
            weights.append(origin_weights[new_dict[v]])
    weights = np.array(weights, dtype=float)
    print(weights.shape, padding_count)
    return weights


def mce_trans():
    coed_weights = dict()
    vocabs = pd.read_csv(mce_origin_dict_file, header=0).vols.tolist()
    with open(mce_vectors_file) as f:
        for line in f.readlines()[2:]:
            w = [float(e) for e in line.split()]
            index = int(w[0])
            weight = w[1:]
            coed_weights['D_' + vocabs[index]] = weight
    embedding_size = len(weight)
    padding_array = [0] * embedding_size
    weights = []
    padding_count = 0
    for k, v in tesa_dict.items():
        if v not in coed_weights.keys():
            weights.append(padding_array)
            padding_count += 1
        else:
            weights.append(coed_weights[v])
    weights = np.array(weights, dtype=float)
    norm1 = weights / np.linalg.norm(weights)
    print(norm1.shape, padding_count)
    print(vocabs)
    return norm1


def tesan_trans(model_type='tesan'):
    if model_type == 'cbow':
        origin_weights = np.genfromtxt(cbow_file, skip_header=2, delimiter=" ")
        weights = []
        origin_weights = origin_weights[:, 1:101]
        embedding_size = origin_weights.shape[1]
        padding_array = np.zeros(embedding_size)
        weights.append(padding_array)
        for i in range(origin_weights.shape[0]):
            weights.append(origin_weights[i])
        weights = np.array(weights, dtype=float)
        print(weights.shape, model_type)
        return weights
    elif model_type == 'sg':
        origin_weights = np.loadtxt(sg_file, delimiter=",")
        weights = []
        embedding_size = origin_weights.shape[1]
        padding_array = np.zeros(embedding_size)
        weights.append(padding_array)
        for i in range(origin_weights.shape[0]):
            weights.append(origin_weights[i])
        weights = np.array(weights, dtype=float)
        print(weights.shape, model_type)
        return weights
    elif model_type == 'tesan':
        origin_weights = np.loadtxt(tesan_file, delimiter=",")
        print(origin_weights.shape, model_type)
        return origin_weights
    elif model_type == 'multi-sa':
        origin_weights = np.loadtxt(sa_file, delimiter=",")
        print(origin_weights.shape, model_type)
        return origin_weights
    elif model_type == 'interval':
        origin_weights = np.loadtxt(interval_file, delimiter=",")
        print(origin_weights.shape, model_type)
        return origin_weights
    elif model_type == 'normal-sa':
        origin_weights = np.loadtxt(normal_file, delimiter=",")
        print(origin_weights.shape, model_type)
        return origin_weights


if __name__ == '__main__':
    # root_dir, _ = os.path.split(os.path.abspath(__file__))
    # root_dir = os.path.dirname(root_dir)
    # root_dir = os.path.dirname(root_dir)
    # root_dir = os.path.dirname(root_dir)
    #
    # tesa_dict_file = join(root_dir, 'dataset/baselines/TeSAN/mimic3/mimic3_dict.json')
    # with open(tesa_dict_file, 'r', encoding='utf-8') as f:
    #     tesa_dict = json.load(f)

    # vectors_file = join(root_dir, 'dataset/baselines/GloVe/mimic3/mimic3_vectors_6.txt')
    # glove_trans(vectors_file, tesa_dict)

    # vectors_file = join(root_dir, 'dataset/baselines/med2vec/mimic3/outputs_med2vec.9.npz')
    # origin_dict_file = join(root_dir, 'dataset/baselines/med2vec/mimic3/outputs.types')
    # med2vec_trans(vectors_file, origin_dict_file, tesa_dict)

    # vectors_file = join(root_dir, 'dataset/baselines/MCE/mimic3/mimic3-attn1-e10-ne10-w6-aw20_month.vec')
    # origin_dict_file = join(root_dir, 'dataset/baselines/MCE/mimic3/mmimic3_volcabs.csv')
    # mce_trans(vectors_file, origin_dict_file, tesa_dict)

    # cbow_file = join(root_dir, 'dataset/baselines/CBOW/mimic3/cbow_sk_6_epoch_10.vect')
    # sg_file = join(root_dir, 'dataset/baselines/SG/mimic3/sg_sk_6_epoch_10.vect')
    # tesa_file = join(root_dir, 'dataset/baselines/TeSAN/mimic3/mimic3_model_tesa_epoch_30_sk_6.vect')

    tesan_trans('cbow')
    # tesan_trans('sg')
    # tesan_trans(tesa_file, model_type='tesa')
