# 美化错误输出 --------------------------------------------------------------------------------------
import inspect
from rich.traceback import install
install()
# -------------------------------------------------------------------------------------------------

# timer 装饰器 -------------------------------------------------------------------------------------
def timing_decorator(func):
  import time
  from functools import wraps  # 导入 wraps 装饰器
  @wraps(func)  # 使用 wraps 装饰器来保留原始函数的元数据
  def wrapper(*args, **kwargs):
    start_time = time.time()  # 记录开始时间
    result = func(*args, **kwargs)  # 执行被装饰的函数
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算执行时间
    BOLD, RESET, lightred = '\033[1m', '\033[0m', '\033[91m'  # ANSI 转义码来设置文本颜色
    # 打印带有颜色和样式的运行时间
    print(f"{lightred}{BOLD}[INFO] {func.__name__} executed in {execution_time:.4f}s{RESET} ")
    return result
  return wrapper
# -------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from torch import logit, nn
from torch.nn import functional as F
import torch
import math
import os
import time
# -------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

  def __init__(self, config):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    # key, query, value projections for all heads, but in a batch
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
    # output projection
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1
    # regularization
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                   .view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    """
    计算一个batch的多头QKV, 头维度移动到batch维度
    nh="number of heads", hs="head size", and C (number of channels) = nh * hs
    例如GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
    """
    B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    
    ## attention (materializes the large (T,T) matrix for all the queries and keys)
    # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
    # att = F.softmax(att, dim=-1)
    # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  #$ Flash Attention, 未采用dropout
    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
    # output projection
    y = self.c_proj(y)
    return y

class MLP(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.c_fc  = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.gelu  = nn.GELU(approximate='tanh')
    self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1

  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

class Block(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

@dataclass
class GPTConfig:
  block_size: int = 1024 # max sequence length
  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
  vocab_size: int = 50257 
  n_layer: int = 12 # number of layers
  n_head: int = 12 # number of heads
  n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

  def __init__(self, config):
    super().__init__()
    self.config = config

    self.transformer = nn.ModuleDict(dict(
      wte = nn.Embedding(config.vocab_size, config.n_embd),
      wpe = nn.Embedding(config.block_size, config.n_embd),
      h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
      ln_f = nn.LayerNorm(config.n_embd),
    ))
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    # weight sharing scheme
    self.transformer.wte.weight = self.lm_head.weight

    # init params
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      std = 0.02
      if hasattr(module, 'NANOGPT_SCALE_INIT'):
        std *= (2 * self.config.n_layer) ** -0.5
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    """计算前向传播；如果targets给定，则额外计算损失
    idx shape (B, T), targets shape (B, T)
    """
    B, T = idx.size() 
    assert T <= self.config.block_size, \
      f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
    #@ 取token嵌入和位置嵌入，相加
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # (T,)
    pos_emb = self.transformer.wpe(pos) # (T, n_embd)
    tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
    x = tok_emb + pos_emb  # (B, T, n_embd)
    #@ 前向传播所有transformer块
    for block in self.transformer.h:
      x = block(x)
    #@ 前向传播最后一层layernorm和分类层
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x) #@ (B, T, vocab_size)
    loss = None
    if targets is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) 
    return logits, loss

  @classmethod
  def from_pretrained(cls, model_type):
    """Loads pretrained GPT-2 model weights from huggingface"""
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel
    print("loading weights from pretrained gpt: %s" % model_type)

    # 模型类型决定 n_layer, n_head, n_embd
    config_args = {
      'gpt2':     dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
      'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
      'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
      'gpt2-xl':    dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    config_args['vocab_size'] = 50257 # 对于gpt checkpoints 都是50257
    config_args['block_size'] = 1024  # 对于gpt checkpoints 都是1024
    # 从头开始初始化 miniGPT 模型
    config = GPTConfig(**config_args)
    model = GPT(config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # 去掉这种mask/buffer, 因为没有参数

    #@ 加载HF/transfomers的GPT2模型, 并把参数拷贝到我们的模型
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    ## 确保所有参数对齐且名称和形状匹配的情况下进行复制
    sd_keys_hf = sd_hf.keys()   # key
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # 忽略buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # 忽略mask（buffer）
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 
                  'mlp.c_proj.weight']  # 需要转置的层
    ## openai的checkpoint用的是Conv1D层，当前用的是Linear层，所以需要转置
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
    for k in sd_keys_hf:  # 遍历从HF加载的模型参数名称
      if any(k.endswith(w) for w in transposed):  # 如果k以transposed中的任何一个结尾则转置
        assert sd_hf[k].shape[::-1] == sd[k].shape # 访问sd[k],如果k不在我们的模型中会报错 
        with torch.no_grad():  # 防止跟踪梯度
          sd[k].copy_(sd_hf[k].t())  # 修改sd[k]为sd_hf[k]的转置，原地修改_
      else:
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
          sd[k].copy_(sd_hf[k])
    return model

  
  def configure_optimizers(self, weight_decay, learning_rate, device_type):
    """配置优化器。
    根据参数维度将参数分为两组，分别应用不同的weight decay（L2正则化）。控制台输出需要优化的参数大小。
    创建并返回AdamW优化器，若设备为CUDA且支持，使用fused版本的AdamW。
    
    注意：参数设置参考gpt3论文（gpt2论文没给参数，gpt2模型架构和gpt3类似）。
    """
    # 获取所有需要梯度更新的参数
    param_dict = {pn: p for pn, p in self.named_parameters()}  # .named_parameters() 返回参数名+参数本身
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # 挑选需要梯度的
    # 创建需要decay的参数组：所有>=2维的张量（例如权重，embedding）；低于2维则nodecay（例如biases，LayerNorm）
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    # 设置参数组字典，之后传给AdamW
    optim_groups = [
      {'params':decay_params, 'weight_decay':weight_decay},
      {'params':nodecay_params, 'weight_decay': 0.0},
    ]
    # 计算两组需要优化的参数量
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}", \
            f"with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}", \
            f"with {num_nodecay_params:,} parameters")
    # 检查能否使用fused版本的AdamW优化器并尝试使用 （fused指的是kernel融合，提高计算性能）
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda" # fused只支持cuda
    use_fused=False
    print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(
      optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer

#% ------------------------------------------------------------------------------------------------
import tiktoken

class DataLoaderLite:
  def __init__(self, B, T, grad_accum_steps=1):
    self.B = B
    self.T = T

    # 加载数据到内存（之后要改数据）
    with open('input.txt', 'r') as f:
      text = f.read()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(text)
    self.tokens = torch.tensor(tokens)
    print(f"loaded {len(self.tokens)} tokens")
    print(f"with grad accum 1 epoch = {len(self.tokens) // (B * T * grad_accum_steps)} batches")
    # 整个 tinyShakespeare 一共 338025 个token。 524288>338025
    self.current_position = 0 # state

  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position : self.current_position+B*T+1]
    x = (buf[:-1]).view(B, T) #@ inputs（idx）
    y = (buf[1:]).view(B, T) #@ targets
    self.current_position += B * T # 前进一个Batch步长
    # 如果下一个位置超过末端，就直接归0。BUG 存在最后一部分数据永远无法访问到的情况。
    if self.current_position + (B * T + 1) > len(self.tokens):
      self.current_position = 0
    return x, y

#% ------------------------------------------------------------------------------------------------
#@ 自动检测设备
device = "cpu"
device_type = "cpu"
if torch.cuda.is_available():
  device = "cuda"
  device_type = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
  device = "mps"
print(f"using device: {device}")

#@ 设置种子
torch.manual_seed(1337)
if torch.cuda.is_available():
  torch.cuda.manual_seed(1337)

#@ 设置批量大小
total_batch_size = 524288 # 2^19, ~0.5M, 单位是token的数量
B = 4 # mirco batch size， 多个微批组成一批，因为要累积梯度 (作者的B用的是16)
T = 1024
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


train_loader = DataLoaderLite(B=B, T=T, grad_accum_steps=grad_accum_steps)

torch.set_float32_matmul_precision('high')  #@ 设置使用 tf32

#@ 创建模型
model = GPT(GPTConfig(vocab_size=50304))  # 50304=2^7*393，能被许多2的幂整除，性能会更好
model.to(device)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
  """学习率调整"""
  # 1. 如果当前的训练步数（it）小于预热步数（warmup_steps），则进行线性预热
  if it < warmup_steps:
    return max_lr * (it + 1) / warmup_steps  # +1 是防止第一步学习率为0
  # 2. 如果当前训练步数大于最大步数（max_steps），则直接返回最小学习率（min_lr）
  if it > max_steps:
    return min_lr
  # 3. 如果当前训练步数处于预热和最大步数之间，则使用余弦衰减的方式计算学习率，该系数会随着训练的进展从1逐渐衰减到0
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  
  assert 0 <= decay_ratio <= 1
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  # 系数从1衰减到0，因此最终学习率将从最大值衰减到最小值
  return min_lr + coeff * (max_lr - min_lr)
  

model.compile()  #@ 编译模型

#@ 创建优化器
# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

#@ 训练
t_st = time.time()
for step in range(max_steps):
  t0 = time.time()
  optimizer.zero_grad()
  loss_accum = 0.0
  for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
      logits, loss = model(x, y)
    # cross_entropy 默认reduction=mean，损失计算公式为L/B。
    # 梯度累积的情况下使用小批量B/gas，也就是L*gas/B，还需要除以一个gas。
    loss /= grad_accum_steps
    loss_accum += loss.detach()
    loss.backward()
    # import code; code.interact(local=locals())
  # 控制全梯度L2范数不超过1.0；返回原始全梯度的L2范数
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  
  lr = get_lr(step)  # 获取学习率
  for param_group in optimizer.param_groups:  # 需手动更新每一个参数组的学习率
    param_group['lr'] = lr 
  optimizer.step()
  torch.cuda.synchronize(device) # cpu给gpu发指令，很快就会运行到下面，但此时gpu还没有执行完。于是手动等待同步
  t1 = time.time()
  dt = t1 - t0
  tokens_processed = train_loader.B * train_loader.T * grad_accum_steps  # 524288
  tokens_per_sec = tokens_processed / dt
  print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} |", \
        f"dt: {dt*1000:.2f}ms | tok/sec {tokens_per_sec:.2f}")
  
t_ed = time.time()
print(f"total training time: {t_ed - t_st:.2f}s")
print(f"tok/sec: {50*train_loader.B*train_loader.T / (t_ed - t_st):.2f}")
import sys; sys.exit(0)

#$ ------------------------------------------------------------------------------------------------
# prefix tokens
model = GPT.from_pretrained('gpt2')
model.to(device)  # 不需要model=model.to(device)
model.eval()
num_return_sequences = 5
max_length = 30

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
  # forward the model to get the logits
  with torch.no_grad():
    output = model(x) # (B, T, vocab_size)
    logits = output[0] if isinstance(output, tuple) else output  # 一个或两个返回值
    # take the logits at the last position
    logits = logits[:, -1, :] # (B, vocab_size)
    # get the probabilities
    probs = F.softmax(logits, dim=-1)
    # do top-k sampling of 50 (huggingface pipeline default)
    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    # select a token from the top-k probabilities
    # note: multinomial does not demand the input to sum to 1
    ix = torch.multinomial(topk_probs, 1) # (B, 1)
    # gather the corresponding indices
    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
    # append to the sequence
    x = torch.cat((x, xcol), dim=1)

# print the generated text
for step in range(num_return_sequences):
  tokens = x[step, :max_length].tolist()
  decoded = enc.decode(tokens)
  print(">", decoded)
