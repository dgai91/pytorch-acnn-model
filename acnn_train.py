import attention.data_pro as pro
import attention.pyt_att as pa
import torch.utils.data as D
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
DW = 100
N = 123
DP = 20
NP = 123
NR = 19
DC = 200
KP = 0.5
K = 3
LR = 0.3
BATCH_SIZE = 300
epochs = 1000
data = pro.load_data('train.txt')
word_dict = pro.build_dict(data[0])
x, y, e1, e2, dist1, dist2 = pro.vectorize(data, word_dict, N)

model = pa.ACNN(N, DW, DP, NP, len(word_dict) + 1, K, NR, DC, KP).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)  # optimize all rnn parameters
loss_func = pa.NovelDistanceLoss(NR)

x = torch.from_numpy(x.astype(np.int64))
y = torch.LongTensor(y)
e1 = torch.LongTensor(e1).view(-1, 1)
e2 = torch.LongTensor(e2).view(-1, 1)
dist1 = torch.LongTensor(dist1)
dist2 = torch.LongTensor(dist2)
x_cat = torch.cat([x, e1, e2, dist1, dist2], 1)
datasets = D.TensorDataset(data_tensor=x_cat, target_tensor=y)
dataloader = D.DataLoader(datasets, BATCH_SIZE, True, num_workers=1)


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
    return acc * 100


for i in range(epochs):
    for step, (b_x_cat, b_y) in enumerate(dataloader):
        list_x = np.split(b_x_cat.numpy(), [N, N + 1, N + 2, N + 2 + NP], 1)
        bx = Variable(torch.from_numpy(list_x[0])).cuda()
        be1 = Variable(torch.from_numpy(list_x[1])).cuda()
        be2 = Variable(torch.from_numpy(list_x[2])).cuda()
        bd1 = Variable(torch.from_numpy(list_x[3])).cuda()
        bd2 = Variable(torch.from_numpy(list_x[4])).cuda()
        by = Variable(b_y).cuda()
        wo, rel_weight = model(bx, be1, be2, bd1, bd2, by) 
        acc = prediction(wo, rel_weight, by, NR)

        loss = loss_func(wo, rel_weight, by)  
        print('acc:', acc.cpu().data.numpy()[0], '%   loss:', loss.cpu().data.numpy()[0])
        optimizer.zero_grad() 
        loss.backward()  
        optimizer.step()  
