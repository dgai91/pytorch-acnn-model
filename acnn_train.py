import attention.data_pro as pro
import attention.pyt_att as pa
import torch.utils.data as D
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import KFold

DW = 100
N = 123
DP = 25
NP = 123
NR = 19
DC = 1000
KP = 0.6
K = 3
LR = 0.2
BATCH_SIZE = 50
epochs = 100
data = pro.load_data('/home/dm/adg/remotetest/attention/train.txt')
t_data = pro.load_data('/home/dm/adg/remotetest/attention/test.txt')
word_dict = pro.build_dict(data[0])
x, y, e1, e2, dist1, dist2 = pro.vectorize(data, word_dict, N)
y = np.array(y).astype(np.int64)
np_cat = np.concatenate((x, np.array(e1).reshape(-1, 1), np.array(e2).reshape(-1, 1), np.array(dist1), np.array(dist2)),
                        1)

tx, ty, te1, te2, td1, td2 = pro.vectorize(t_data, word_dict, N)
embed_file = '/home/dm/adg/remotetest/attention/embeddings.txt'
vac_file = '/home/dm/adg/remotetest/attention/words.lst'
embedding = pro.load_embedding(embed_file, vac_file, word_dict)

model = pa.ACNN(N, embedding, DP, NP, K, NR, DC, KP).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=0.0001)  # optimize all rnn parameters
loss_func = pa.NovelDistanceLoss(NR)


def data_unpack(cat_data, target):
    list_x = np.split(cat_data.numpy(), [N, N + 1, N + 2, N + 2 + NP], 1)
    bx = Variable(torch.from_numpy(list_x[0])).cuda()
    be1 = Variable(torch.from_numpy(list_x[1])).cuda()
    be2 = Variable(torch.from_numpy(list_x[2])).cuda()
    bd1 = Variable(torch.from_numpy(list_x[3])).cuda()
    bd2 = Variable(torch.from_numpy(list_x[4])).cuda()
    target = Variable(target).cuda()
    return bx, be1, be2, bd1, bd2, target


def prediction(wo, rel_weight, y, NR):
    wo_norm = F.normalize(wo)
    bz = wo_norm.data.size()[0]
    dc = wo_norm.data.size()[1]
    wo_norm_tile = wo_norm.view(bz, 1, dc).repeat(1, NR, 1)
    batched_rel_w = F.normalize(rel_weight).view(1, NR, dc).repeat(bz, 1, 1)
    all_distance = torch.norm(wo_norm_tile - batched_rel_w, 2, 2)
    predict = torch.min(all_distance, 1)[1].long()
    # print(predict)
    correct = torch.eq(predict, y)
    # print(correct)
    acc = correct.sum().float() / float(correct.data.size()[0])
    return (acc * 100).cpu().data.numpy()[0]


k_fold = KFold(n_splits=10)

for i in range(epochs):
    split = 0
    for train_indices, test_indices in k_fold.split(np_cat):
        acc = 0
        loss = 0
        train = torch.from_numpy(np_cat[train_indices].astype(np.int64))
        y_tensor = torch.LongTensor(y[train_indices])
        train_datasets = D.TensorDataset(data_tensor=train, target_tensor=y_tensor)
        train_dataloader = D.DataLoader(train_datasets, BATCH_SIZE, True, num_workers=2)
        j = 0
        for (b_x_cat, b_y) in train_dataloader:
            bx, be1, be2, bd1, bd2, by = data_unpack(b_x_cat, b_y)
            wo, rel_weight = model(bx, be1, be2, bd1, bd2)
            acc += prediction(wo, rel_weight, by, NR)
            l = loss_func(wo, rel_weight, by)
            j += 1
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            loss += l
        split += 1
        print('epoch:', i, 'split:', split, 'acc:', acc / j, '%   loss:', loss.cpu().data.numpy()[0] / j)
        test = torch.from_numpy(np_cat[test_indices].astype(np.int64))
        ty_tensor = torch.LongTensor(y[test_indices])
        test_datasets = D.TensorDataset(data_tensor=test, target_tensor=ty_tensor)
        test_dataloader = D.DataLoader(train_datasets, BATCH_SIZE, True, num_workers=2)
        test_acc = 0
        ti = 0
        t_loss = 0
        for (b_tx_cat, b_ty) in test_dataloader:
            btx, bte1, bte2, btd1, btd2, b_ty = data_unpack(b_tx_cat, b_ty)
            two, t_rel_weight = model(btx, bte1, bte2, btd1, btd2, is_training=False)
            test_acc += prediction(two, t_rel_weight, b_ty, NR)
            # tl = loss_func(two, t_rel_weight, b_ty)
            # t_loss += tl
            ti += 1
        print('epoch:', i, 'split:', split, 'test_acc:', test_acc / ti, '%')#   loss:', t_loss.cpu().data.numpy()[0] / ti)
