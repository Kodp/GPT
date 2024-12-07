#@ 还是得手写手抄才能弄的更清楚、发现自己有许多还不懂的地方。
import torch
import torch.nn as nn
from torch.nn import functional as F
# 美化错误输出 --------------------------------------------------------------------------------------
from rich.traceback import install
install()
#--------------------------------------------------------------------------------------------------

#@ 0. 超参数
BATCH_SIZE    = 64               # 批大小
MAX_CONTEXT_LEN   = block_size = 256 # 上下文长度（注意力矩阵NxN,N的大小）
MAX_ITERS     = 5000 # 一共迭代次数
EVAL_INTERVAL = 500  # 每多少次做一个评估
EVAL_ITERS    = 200  # 每次评估算多少样本（在训练集、测试集上都算，实际计算x2）
LEARNING_RATE = 3e-4 # 学习率
N_EMBD        = 384  # embedding 维度， token嵌入的向量长度
N_HEAD        = 6    # 注意力头数
N_LAYER       = 6    # Block的数量/transformer 堆叠层数
DROPOUT       = 0.2  # dropout 比例

GPU_INDEX     = 0
DEVICE        = f'cuda:{GPU_INDEX}' if torch.cuda.is_available() else 'cpu'

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

print("vocab size:", vocab_size)

encode = lambda text_seq: [stoi[c] for c in text_seq]
decode = lambda idx_seq: ''.join([itos[i] for i in idx_seq])


#@ 划分集合, 读取数据
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


model = None

def get_batch(split:str):
  """返回一批数据。split:指示训练集或测试集"""
  data = train_data if split == 'train' else val_data
  # 从data中随机采样batch_size个值，每个值的范围在[0, len(data) - MAX_CONTEXT_LEN)，保证不越界
  start_ix = torch.randint(len(data) - MAX_CONTEXT_LEN, (BATCH_SIZE, ))  # (a, ) 表示元组，参数要求元组
  x  = torch.stack([data[i:i+MAX_CONTEXT_LEN] for i in start_ix])     # (B, max_context_len)
  y  = torch.stack([data[i+1:i+MAX_CONTEXT_LEN+1] for i in start_ix]) # (B, max_context_len)
  x, y = x.to(DEVICE), y.to(DEVICE)
  return x, y


#@ 评估损失
@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()

  for split in ['train', 'val']:
    losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  
  model.train()
  return out

#@ 模型
class Head(nn.Module):
  """单头自注意力"""
  def __init__(self, head_size):
    super().__init__()
    self.key   = nn.Linear(N_EMBD, head_size, bias=False)
    self.query = nn.Linear(N_EMBD, head_size, bias=False)
    self.value = nn.Linear(N_EMBD, head_size, bias=False)
    # `tril` 下三角全为1，其余为0，用于确保模型只基于过去的token预测下一个token；buffer张量不参与梯度计算；
    self.register_buffer('tril', torch.tril(torch.ones(MAX_CONTEXT_LEN, MAX_CONTEXT_LEN))) 
    #$ tril矩阵在每个HEAD里都有一个拷贝，导致内存浪费。可以放到MultiHeadAttention或GPTLanguageModel里。
    self.dropout = nn.Dropout(DROPOUT)
  
  def forward(self, x):
    """
    x.shape:(Batch, Time-step, Channel); 
    输出.shape:(Batch, Time-step, Channel)
    """
    B, T, C = x.shape  #@ T <= context_len
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
    self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    #@ proj 保证了多头输出的维度和输入维度一致，把head_size * num_heads映射到 N_EMBD
    self.proj    = nn.Linear(head_size * num_heads, N_EMBD)  # 外部 N_EMBD
    self.dropout = nn.Dropout(DROPOUT)
  
  def forward(self, x):
    """x:(B, T, C) -> (B, T, C), C=n_embd"""
    # 每个多头的输出的维度(T,head_size),cat之后Y的维度是(T,head_size*n_head)
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))  # 经过proj后，(T,head_size*n_head) -> (T,N_EMBD)
    return out
    
    
class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4 * n_embd),
      nn.ReLU(),
      nn.Linear(4 * n_embd, n_embd),
      nn.Dropout(DROPOUT),
    )
  def forward(self, x):
    return self.net(x)


# (B, T, C) -> (B, T, C), C = n_embd
class Block(nn.Module):
  """Transformer"""
  def __init__(self, n_embd, n_head):
    """n_embd:嵌入向量长度  n_head: 注意力头数"""
    super().__init__()
    head_size = n_embd // n_head  # 凑整；实际上这里可以自由设置
    self.sa   = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1  = nn.LayerNorm(n_embd)
    self.ln2  = nn.LayerNorm(n_embd)
  
  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
  
class GPTLanguageModel(nn.Module):
    
  def _init_weights(self, module):
    """更好的初始化"""
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, N_EMBD)
    self.position_embedding_table = nn.Embedding(MAX_CONTEXT_LEN, N_EMBD)
    self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
    self.ln_f = nn.LayerNorm(N_EMBD)  # final layer norm
    self.lm_head = nn.Linear(N_EMBD, vocab_size)  # 映射最后一维n_embd大小到vocab_size，用于最后采样token
    
    # 初始化, apply(func) 遍历所有子模块,将func作用在每个子模块上
    self.apply(self._init_weights)
   
  def forward(self, idx, targets=None):
    """idx.shape=(Batch size, Time-step),值为token"""
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) # 得到token的嵌入 (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T,C)
    x       = tok_emb + pos_emb  # (B,T,C)
    x       = self.blocks(x)     # (B,T,C)
    x       = self.ln_f(x)       # (B,T,C)
    logits  = self.lm_head(x)    # (B,T,vocab_size)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)
    # 一般loss不在forward里面计算，而是在外部计算，这里为了方便
    return logits, loss
    
  def generate(self, idx, max_new_tokens):
    """idx.shape=(B,T),是当前上下文的token"""
    for _ in range(max_new_tokens):
      # 截断，只保留最近的max_context_len个token
      idx_cond = idx[:, -MAX_CONTEXT_LEN:] # (B,T<=max_context_len); 这里可以取的比MAX_CONTEXT_LEN小
      # 前向传播
      logits, loss = self(idx_cond) # (B,T,C)
      # 只看最后一步
      logits   = logits[:, -1, :]   # (B,C)
      probs    = F.softmax(logits, dim=-1) # (B,C)
      # 从预测的分布中采样一个token的idx
      idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
      # 添加一个idx到上下文索引idx
      idx      = torch.cat((idx, idx_next), dim=1)  # (B,T+1) !注意这里是在时间维度上拼接,而不是在batch维度上拼接

    return idx


#@ 训练+输出
model = GPTLanguageModel()
m = model.to(DEVICE)
# 输出模型参数量
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# 提示开始训练
print("Start training...") 
for iter in range(MAX_ITERS):
  if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
    losses = estimate_loss()
    print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  

context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
  