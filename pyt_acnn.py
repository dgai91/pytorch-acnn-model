import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from sklearn import preprocessing
import torch.nn.functional as F


def l2_normalize(value):
    array = value.data.numpy()
    norm = preprocessing.normalize(array, norm='l2')
    var_norm = Variable(torch.from_numpy(norm), requires_grad=True)
    return var_norm


def tile(value, multiples, axis=1):
    array = value.data.numpy().astype(float)
    shape = array.shape
    if axis == 1:
        array = array.reshape(shape[0], 1, shape[1])
    elif axis == 2:
        array = array.reshape(shape[0], shape[1], 1)
    elif axis == 0:
        array = array.reshape(1, shape[0], shape[1])
    array = array.repeat(multiples, axis)
    return Variable(torch.FloatTensor(array), requires_grad=True)


def triple_dim_mm(mat1, mat2):
    np_mat1 = mat1.data.numpy()
    np_mat2 = mat2.data.numpy()
    mat = np.matmul(np_mat1, np_mat2)
    torch_mat = Variable(torch.FloatTensor(mat), requires_grad=True)
    return torch_mat


def one_hot(indices, depth, on_value=1, off_value=0):
    np_ids = np.array(indices.data.numpy()).astype(int)
    if len(np_ids.shape) == 2:
        encoding = np.zeros([np_ids.shape[0], np_ids.shape[1], depth], dtype=int)
        added = encoding + off_value
        for i in range(np_ids.shape[0]):
            for j in range(np_ids.shape[1]):
                added[i, j, np_ids[i, j]] = on_value
        return Variable(torch.FloatTensor(added.astype(float)), requires_grad=True)
    if len(np_ids.shape) == 1:
        encoding = np.zeros([np_ids.shape[0], depth], dtype=int)
        added = encoding + off_value
        for i in range(np_ids.shape[0]):
            added[i, np_ids[i]] = on_value
        return Variable(torch.FloatTensor(added.astype(float)), requires_grad=True)


class ACNN(nn.Module):
    def __init__(self, max_len, embedding_size, pos_embed_size,
                 pos_embed_num, vac_size, slide_window,
                 class_num, num_filters, keep_prob, is_training=True):
        super(ACNN, self).__init__()

        self.n = max_len
        self.dw = embedding_size
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.p = (self.k - 1) // 2

        self.e1_embedding = nn.Embedding(vac_size, embedding_size)
        self.e2_embedding = nn.Embedding(vac_size, embedding_size)
        self.x_embedding = nn.Embedding(vac_size, embedding_size)
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)
        self.y_embedding = nn.Embedding(self.nr, self.dc)
        self.dropout = nn.Dropout(self.keep_prob)
        self.conv = nn.Conv2d(1, self.dc, (self.k, self.d), (1, self.d), (self.p, 0))
        self.tanh = nn.Tanh()
        self.U = nn.Parameter(torch.FloatTensor(self.dc, self.nr))
        self.max_pool = nn.MaxPool2d((1, self.dc), (1, self.dc))
        self.softmax = nn.Softmax()

    def input_attention(self, x, e1, e2):
        bz = x.data.size()[0]
        x_embed = self.x_embedding(x)
        e1_embed = self.e1_embedding(e1)
        e2_embed = self.e2_embedding(e2)
        A1 = triple_dim_mm(x_embed, e1_embed.view(bz, self.dw, 1))
        A2 = triple_dim_mm(x_embed, e2_embed.view(bz, self.dw, 1))
        A1 = A1.view(bz, self.n)
        A2 = A2.view(bz, self.n)
        alpha1 = self.softmax(A1)
        alpha2 = self.softmax(A2)
        alpha = torch.div(torch.add(alpha1, alpha2), 2)
        return alpha, x_embed

    def convolution(self, x_embed, dist1, dist2, alpha):
        bz = x_embed.data.size()[0]
        dist1_embed = self.dist1_embedding(dist1)
        dist2_embed = self.dist2_embedding(dist2)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed), 2)
        x_concat = x_concat.view(bz, 1, self.n, self.d)
        print(x_concat.data.size())
        R = self.conv(x_concat)
        R = self.tanh(R).view(bz, self.n, self.dc)
        alpha = tile(alpha, self.dc, 2)
        R = torch.mul(R, alpha)
        return R  # bz, n, dc

    def attentive_pooling(self, R, in_y):
        y_embed = self.y_embedding(in_y)
        bz = y_embed.data.size()[0]
        rel_weight = self.y_embedding.weight
        G = torch.mm(R.view(bz * self.n, self.dc), self.U)  # (bz*n, nr)
        G = torch.mm(G, rel_weight)  # (bz*n, dc)
        AP = F.softmax(G)
        AP = AP.view(bz, self.n, self.dc)
        wo = triple_dim_mm(torch.transpose(R, 2, 1), AP)  # bz, dc, dc
        wo = self.max_pool(wo.view(bz, 1, self.dc, self.dc))
        return wo.view(bz, self.dc), rel_weight

    def forward(self, x, e1, e2, dist1, dist2, y):
        alpha, x_embed = self.input_attention(x, e1, e2)
        R = self.convolution(x_embed, dist1, dist2, alpha)
        wo, rel_weight = self.attentive_pooling(R, y)
        return wo, rel_weight


class NovelDistanceLoss(nn.Module):
    def __init__(self, nr, margin=1):
        super(NovelDistanceLoss, self).__init__()
        self.nr = nr
        self.margin = margin

    def forward(self, wo, rel_weight, in_y, epsilon=1e-12):
        wo_norm = l2_normalize(wo)  # (bz, dc)
        bz = wo_norm.data.size()[0]
        wo_norm_tile = tile(wo_norm, self.nr)  # (bz, nr, dc)
        batched_rel_w = tile(l2_normalize(rel_weight), wo_norm.data.size()[0], 0)
        all_distance = torch.norm(wo_norm_tile - batched_rel_w, 2, 2)  # (bz, nr, 1)
        mask = one_hot(in_y, self.nr, 1000, 0)  # (bz, nr)
        masked_y = torch.add(all_distance.view(bz, self.nr), mask)
        neg_y = np.argmin(masked_y.data.numpy(), 1).astype(np.int64)  # (bz,)
        neg_y = Variable(torch.from_numpy(neg_y), requires_grad=True)
        neg_y = torch.mm(one_hot(neg_y, self.nr), rel_weight)  # (bz, nr)*(nr, dc) => (bz, dc)
        pos_y = torch.mm(one_hot(in_y, self.nr), rel_weight)
        neg_distance = torch.norm(wo_norm - l2_normalize(neg_y), 2, 1)
        pos_distance = torch.norm(wo_norm - l2_normalize(pos_y), 2, 1)
        loss = torch.mean(pos_distance + (self.margin - neg_distance))
        print('ok')
        return loss


