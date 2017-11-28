import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


def one_hot(indices, depth, on_value=1, off_value=0):
    np_ids = np.array(indices.cpu().data.numpy()).astype(int)
    if len(np_ids.shape) == 2:
        encoding = np.zeros([np_ids.shape[0], np_ids.shape[1], depth], dtype=int)
        added = encoding + off_value
        for i in range(np_ids.shape[0]):
            for j in range(np_ids.shape[1]):
                added[i, j, np_ids[i, j]] = on_value
        return Variable(torch.FloatTensor(added.astype(float))).cuda()
    if len(np_ids.shape) == 1:
        encoding = np.zeros([np_ids.shape[0], depth], dtype=int)
        added = encoding + off_value
        for i in range(np_ids.shape[0]):
            added[i, np_ids[i]] = on_value
        return Variable(torch.FloatTensor(added.astype(float))).cuda()


class ACNN(nn.Module):
    def __init__(self, max_len, embedding, pos_embed_size,
                 pos_embed_num, slide_window, class_num,
                 num_filters, keep_prob):
        super(ACNN, self).__init__()
        self.dw = embedding.shape[1]
        self.vac_len = embedding.shape[0]
        self.dp = pos_embed_size
        self.d = self.dw + 2 * self.dp
        self.np = pos_embed_num
        self.nr = class_num
        self.dc = num_filters
        self.keep_prob = keep_prob
        self.k = slide_window
        self.p = (self.k - 1) // 2
        self.n = max_len
        self.kd = self.d * self.k
        self.e1_embedding = nn.Embedding(self.vac_len, self.dw)
        self.e1_embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.e2_embedding = nn.Embedding(self.vac_len, self.dw)
        self.e2_embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.x_embedding = nn.Embedding(self.vac_len, self.dw)
        self.x_embedding.weight = nn.Parameter(torch.from_numpy(embedding))
        self.dist1_embedding = nn.Embedding(self.np, self.dp)
        self.dist2_embedding = nn.Embedding(self.np, self.dp)
        self.pad = nn.ConstantPad2d((0, 0, self.p, self.p), 0)
        self.y_embedding = nn.Embedding(self.nr, self.dc)
        self.dropout = nn.Dropout(self.keep_prob)
        # self.conv = nn.Conv2d(1, self.dc, (self.k, self.kd), (1, self.kd), (self.p, 0), bias=True)
        self.conv = nn.Conv2d(1, self.dc, (1, self.kd), (1, self.kd), bias=True)  # renewed
        self.tanh = nn.Tanh()
        self.U = nn.Parameter(torch.randn(self.dc, self.nr))
        self.We1 = nn.Parameter(torch.randn(self.dw, self.dw))
        self.We2 = nn.Parameter(torch.randn(self.dw, self.dw))
        self.max_pool = nn.MaxPool2d((1, self.dc), (1, self.dc))
        self.softmax = nn.Softmax()

    def window_cat(self, x_concat):
        s = x_concat.data.size()
        px = self.pad(x_concat.view(s[0], 1, s[1], s[2])).view(s[0], s[1] + 2 * self.p, s[2])
        t_px = torch.index_select(px, 1, Variable(torch.LongTensor(range(s[1]))).cuda())
        m_px = torch.index_select(px, 1, Variable(torch.LongTensor(range(1, s[1] + 1))).cuda())
        b_px = torch.index_select(px, 1, Variable(torch.LongTensor(range(2, s[1] + 2))).cuda())
        return torch.cat([t_px, m_px, b_px], 2)

    def new_input_attention(self, x, e1, e2, dist1, dist2, is_training=True):
        bz = x.data.size()[0]
        x_embed = self.x_embedding(x) # (bz, n, dw)
        e1_embed = self.e1_embedding(e1)
        e2_embed = self.e2_embedding(e2)
        dist1_embed = self.dist1_embedding(dist1)
        dist2_embed = self.dist2_embedding(dist2)
        x_concat = torch.cat((x_embed, dist1_embed, dist2_embed), 2)
        w_concat = self.window_cat(x_concat)
        if is_training:
            w_concat = self.dropout(w_concat)
        W1 = self.We1.view(1, self.dw, self.dw).repeat(bz, 1, 1)
        W2 = self.We2.view(1, self.dw, self.dw).repeat(bz, 1, 1)
        W1x = torch.bmm(x_embed, W1)
        W2x = torch.bmm(x_embed, W2)
        A1 = torch.bmm(W1x, e1_embed.view(bz, self.dw, 1))  # (bz, n, 1)
        A2 = torch.bmm(W2x, e2_embed.view(bz, self.dw, 1))
        A1 = A1.view(bz, self.n)
        A2 = A2.view(bz, self.n)
        alpha1 = self.softmax(A1)
        alpha2 = self.softmax(A2)
        alpha = torch.div(torch.add(alpha1, alpha2), 2)
        alpha = alpha.view(bz, self.n, 1).repeat(1, 1, self.kd)
        return torch.mul(w_concat, alpha)

    def new_convolution(self, R):
        s = R.data.size()  # bz, n, k*d
        R = self.conv(R.view(s[0], 1, s[1], s[2]))  # bz, dc, n, 1
        R = self.tanh(R)  # added
        R_star = R.view(s[0], self.dc, s[1])
        return R_star  # bz, dc, n

    def attentive_pooling(self, R_star):
        rel_weight = self.y_embedding.weight
        bz = R_star.data.size()[0]

        b_U = self.U.view(1, self.dc, self.nr).repeat(bz, 1, 1)
        b_rel_w = rel_weight.view(1, self.nr, self.dc).repeat(bz, 1, 1)
        G = torch.bmm(R_star.transpose(2, 1), b_U)  # (bz, n, nr)
        G = torch.bmm(G, b_rel_w)  # (bz, n, dc)
        AP = F.softmax(G)
        AP = AP.view(bz, self.n, self.dc)
        wo = torch.bmm(R_star, AP)  # bz, dc, dc
        wo = self.max_pool(wo.view(bz, 1, self.dc, self.dc))
        return wo.view(bz, 1, self.dc).view(bz, self.dc), rel_weight

    def forward(self, x, e1, e2, dist1, dist2, is_training=True):
        R = self.new_input_attention(x, e1, e2, dist1, dist2, is_training)
        R_star = self.new_convolution(R)
        wo, rel_weight = self.attentive_pooling(R_star)
        wo = F.relu(wo)
        return wo, rel_weight


class NovelDistanceLoss(nn.Module):
    def __init__(self, nr, margin=1):
        super(NovelDistanceLoss, self).__init__()
        self.nr = nr
        self.margin = margin

    def forward(self, wo, rel_weight, in_y):
        wo_norm = F.normalize(wo)  # (bz, dc)
        bz = wo_norm.data.size()[0]
        dc = wo_norm.data.size()[1]
        wo_norm_tile = wo_norm.view(-1, 1, dc).repeat(1, self.nr, 1)  # (bz, nr, dc)
        batched_rel_w = F.normalize(rel_weight).view(1, self.nr, dc).repeat(bz, 1, 1)
        all_distance = torch.norm(wo_norm_tile - batched_rel_w, 2, 2)  # (bz, nr, 1)
        mask = one_hot(in_y, self.nr, 1000, 0)  # (bz, nr)
        masked_y = torch.add(all_distance.view(bz, self.nr), mask)
        neg_y = torch.min(masked_y, dim=1)[1]  # (bz,)
        neg_y = torch.mm(one_hot(neg_y, self.nr), rel_weight)  # (bz, nr)*(nr, dc) => (bz, dc)
        pos_y = torch.mm(one_hot(in_y, self.nr), rel_weight)
        neg_distance = torch.norm(wo_norm - F.normalize(neg_y), 2, 1)
        pos_distance = torch.norm(wo_norm - F.normalize(pos_y), 2, 1)
        loss = torch.mean(pos_distance + self.margin - neg_distance)
        return loss
