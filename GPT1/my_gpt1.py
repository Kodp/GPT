import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size    = 64               # 批大小
context_len   = block_size = 256 # 上下文长度（注意力矩阵NxN,N的大小）
max_iters     = 5000 # 一共迭代次数
eval_interval = 500  # 每多少次做一个评估
eval_iters    = 200  # 每次评估算多少样本（在训练集、测试集上都算，实际计算x2）
learning_rate = 3e-4 # 学习率
n_embd        = 384  # embedding 维度， token嵌入的向量长度
n_head        = 6    # 注意力头数
n_layer       = 6    # transfomer 堆叠层数
dropout       = 0.2  # dropout 比例

gpu_index     = 0
device        = f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu'

