import numpy as np
import logging
from collections import Counter


def load_data(file):
    sentences = []
    relations = []
    e1_pos = []
    e2_pos = []

    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f.readlines():
            line = line.strip().lower().split()
            relations.append(int(line[0]))
            e1_pos.append((int(line[1]), int(line[2])))  # (start_pos, end_pos)
            e2_pos.append((int(line[3]), int(line[4])))  # (start_pos, end_pos)
            sentences.append(line[5:])

    return sentences, relations, e1_pos, e2_pos


def build_dict(sentences):
    word_count = Counter()
    for sent in sentences:
        for w in sent:
            word_count[w] += 1

    ls = word_count.most_common()
    word_dict = {w[0]: index + 1 for (index, w) in enumerate(ls)}
    # leave 0 to PAD
    return word_dict


def load_embedding(emb_file, emb_vocab, word_dict):
    vocab = {}
    with open(emb_vocab, 'r') as f:
        for id, w in enumerate(f.readlines()):
            w = w.strip().lower()
            vocab[w] = id

    f = open(emb_file, 'r')
    embed = f.readlines()

    dim = len(embed[0].split())
    num_words = len(word_dict) + 1
    embeddings = np.random.uniform(-0.01, 0.01, size=(num_words, dim))

    pre_trained = 0
    for w in vocab.keys():
        if w in word_dict:
            embeddings[word_dict[w]] = [float(x) for x in embed[vocab[w]].split()]
            pre_trained += 1
    embeddings[0] = np.zeros(dim)

    logging.info(
        'embeddings: %.2f%%(pre_trained) unknown: %d' % (pre_trained / num_words * 100, num_words - pre_trained))

    f.close()
    return embeddings.astype(np.float32)


def pos(x):
    '''
    map the relative distance between [0, 123)
    '''
    if x < -60:
        return 0
    if x >= -60 and x <= 60:
        return x + 61
    if x > 60:
        return 122


def vectorize(data, word_dict, max_len):
    sentences, relations, e1_pos, e2_pos = data

    # replace word with word-id
    # sents_vec = []
    e1_vec = []
    e2_vec = []

    num_data = len(sentences)
    sents_vec = np.zeros((num_data, max_len), dtype=int)

    logging.debug('data shape: (%d, %d)' % (num_data, max_len))

    for idx, (sent, pos1, pos2) in enumerate(zip(sentences, e1_pos, e2_pos)):
        vec = [word_dict[w] if w in word_dict else 0 for w in sent]
        sents_vec[idx, :len(vec)] = vec

        # # log e1 and e2 if e1 or e2 is a phrase
        # if pos1[0]!=pos1[1] or pos2[0]!=pos2[1]:
        #   s_e1 = ''
        #   for w in sent[pos1[0] : pos1[1]+1]:
        #     s_e1 += w + ' '
        #   s_e2 = ''
        #   for w in sent[pos2[0] : pos2[1]+1]:
        #     s_e2 += w + ' '
        #   logging.debug("%s - %s" % (s_e1, s_e2))

        # # the entire e1 and e2 phrase
        # e1_vec.append(vec[pos1[0] : pos1[1]+1])
        # e2_vec.append(vec[pos2[0] : pos2[1]+1])

        # last word of e1 and e2
        e1_vec.append(vec[pos1[1]])
        e2_vec.append(vec[pos2[1]])

    # compute relative distance
    dist1 = []
    dist2 = []

    for sent, p1, p2 in zip(sents_vec, e1_pos, e2_pos):
        # current word position - last word position of e1 or e2
        dist1.append([pos(p1[1] - idx) for idx, _ in enumerate(sent)])
        dist2.append([pos(p2[1] - idx) for idx, _ in enumerate(sent)])

    return sents_vec, relations, e1_vec, e2_vec, dist1, dist2


