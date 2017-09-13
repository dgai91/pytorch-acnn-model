import attention.data_pro as pro
import attention.pyt_att as pa
from torch.autograd import Variable
import torch
import numpy as np

DW = 100
N = 123
DP = 20
NP = 123
NR = 19
DC = 10
KP = 0.5
K = 3
LR = 0.0003
BATCH_SIZE = 100
epochs = 10
data = pro.load_data('F:/SemEval2010_task8_all_data/train.txt')
word_dict = pro.build_dict(data[0])
x, y, e1, e2, dist1, dist2 = pro.vectorize(data, word_dict, 123)

model = pa.ACNN(N, DW, DP, NP, len(word_dict) + 1, K, NR, DC, KP)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1)  # optimize all rnn parameters
loss_func = pa.NovelDistanceLoss(NR)
x = Variable(torch.from_numpy(x.astype(np.int64)))[7100:]
y = Variable(torch.LongTensor(y))[7100:]
e1 = Variable(torch.LongTensor(e1))[7100:]
e2 = Variable(torch.LongTensor(e2))[7100:]
dist1 = Variable(torch.LongTensor(dist1))[7100:]
dist2 = Variable(torch.LongTensor(dist2))[7100:]
for i in range(epochs):
    wo, rel_weight = model(x, e1, e2, dist1, dist2, y)  # rnn 对于每个 step 的 prediction, 还有最后一个 step 的 h_state
    # !!  下一步十分重要 !!
    loss = loss_func(wo, rel_weight, y)  # cross entropy loss
    print(loss)
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
