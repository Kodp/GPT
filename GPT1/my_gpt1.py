import torch
import torch.nn as nn
from torch.nn import functional as F

#@ 0. 超参数
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

torch.manual_seed(1337)

#@ 数据集
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    
#@ tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda text_seq: [stoi[c] for c in text_seq]
decode = lambda idx_seq: ''.join([itos[i] for i in idx_seq])


#@ 划分集合, 读取数据
data = torch.tensor(encode(text), type=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


model = None

def get_batch(split:str):
  """返回一批数据。split:指示训练集或测试集"""
  data = train_data if split == 'train' else val_data
  # 从data中随机采样batch_size个值，每个值的范围在[0, len(data) - block_size)，保证不越界
  start_ix = torch.randint(len(data) - context_len, (batch_size, ))  # (a, ) 表示元组，参数要求元组
  x  = torch.stack([data[i:i+context_len]] for i in start_ix)
  y  = torch.stack([data[i+1:i+context_len+1]] for i in start_ix)
  x, y = x.to(device), y.to(device)
  return x, y



@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()

  for split in ['train', 'eval']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  
  model.train()
  return out

class Head(nn.Module):
  """单头自注意力"""
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    # `tril` 下三角全为1，其余为0，用于确保模型只基于过去的token预测下一个token；buffer张量不参与梯度计算；
    self.register_buffer('tril', torch.tril(torch.ones(context_len, context_len))) 
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    """x.shape:(Batch, Time-step, Channel); output.shape:(Batch, Time-step, Channel)"""
    B, T, C = x.shape  # T <= context_len
    k = self.key(x)    # (B,T,hs)  「hs=head size」
    q = self.query(x)  # (B,T,hs)
    # (B,T,hs) @ (B,hs,T) -> (B,T,T)
    s = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
    s = s.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
    
    a = F.softmax(s, dim=-1)  # (B,T,T)
    a = self.dropout(a)       # 防止叠加多层过拟合
    v = self.value(x)
    output = a @ v               # (B,T,T) @ (B,T,hs) -> (B,T,hs)
    
    return output
  
  
class MultiHeadAttention(nn.Module):
  """多头自注意力，每个注意力头并行"""
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embd)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out
    
    
  
  

