import os
import math
import time
import inspect
from dataclasses import dataclass
from turtle import back
import torch
import torch.nn as nn
from torch.nn import functional as F

# 美化错误输出 --------------------------------------------------------------------------------------
import inspect
from rich.traceback import install
install()
# -------------------------------------------------------------------------------------------------

#%1 -----------------------------------------------------------------------------------------------

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
    if master_process:
      print(f"num decayed parameter tensors: {len(decay_params)}", \
              f"with {num_decay_params:,} parameters")
      print(f"num non-decayed parameter tensors: {len(nodecay_params)}", \
              f"with {num_nodecay_params:,} parameters")
    # 检查能否使用fused版本的AdamW优化器并尝试使用 （fused指的是kernel融合，提高计算性能）
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda" # fused只支持cuda
    use_fused=False
    if master_process:
      print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(
      optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer

#%2 ------------------------------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename) -> torch.Tensor:
  """从磁盘上加载numpy数据(token)，返回torch tensor"""
  npt = np.load(filename)  # 加载已经预处理好的数据集：gpt2token格式
  npt = npt.astype(np.int32)  # can't convert np.ndarray of type numpy.uint16.  
  ptt = torch.tensor(npt, dtype=torch.long)  #TODO long? 似乎可以优化，毕竟只是token。减少显存占用。
  return ptt
  

class DataLoaderLite:
  def __init__(self, B, T, process_rank, num_processes, split):
    """简易数据加载器。
    B:
    T:
    process_rank:
    num_processes:
    """
    self.B = B
    self.T = T
    self.process_rank = process_rank
    self.num_processes = num_processes
    assert split in {'train', 'val'}  # train or val
    
    # 加载数据到内存（之后要改数据）
    # with open('input.txt', 'r') as f:
    #   text = f.read()
    # enc = tiktoken.get_encoding('gpt2')
    # tokens = enc.encode(text)
    # self.tokens = torch.tensor(tokens)
    
    data_root = "edu_fineweb10B"  #fix 硬编码
    shards = os.listdir(data_root)  # listdir获取路径下所有文件名、文件夹名
    shards = [s for s in shards if split in s]  #$ 选择文件名含有"train"或"val"的数据^^
    shards = sorted(shards)   # 排序（文件名本身有序）
    shards = [os.path.join(data_root, s) for s in shards]  # 拼接为完整路径
    self.shards = shards
    assert len(shards) > 0, f"no shards found for split {split}"
    
    if master_process:  #fix master_process是下面的全局变量
      print(f"found {len(shards)} shards for split {split}")
    
    self.reset()

  def reset(self):
    """重置数据加载器读取数据的位置"""
    self.current_shard = 0
    self.tokens = load_tokens(self.shards[self.current_shard])  # tensor token 数据
    self.current_position = self.B * self.T * self.process_rank  # 当前设备（进程）的访问数据块的起始位置
    self.data_chunk_size = B * T * self.num_processes   # 全部设备（进程）的访问数据页大小；[块 块]=页
    
  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position : self.current_position+B*T+1]
    x = (buf[:-1]).view(B, T) #@ inputs（idx）
    y = (buf[1:]).view(B, T)  #@ targets
    
    # 分片0[分页0[[块0]，[块1],...], 分页1[[块0]，[块1],...], ...]
    # 前进一个「Batch*进程数量」步长。「Batch*进程数量」是全部进程的访问页，每个进程只访问其中一块，避免重复
    self.current_position += self.data_chunk_size  #* 前进一个分页
    # 如果下一个位置超过末端，就前进到下一个分片。BUG 存在最后一部分数据永远无法访问到的情况。
    if self.current_position + (self.data_chunk_size + 1) > len(self.tokens):
      self.current_shard = (self.current_shard + 1) % len(self.shards)  #* 前进一个分片
      self.tokens = load_tokens(self.shards[self.current_shard])
      self.current_position = self.B * self.T * self.process_rank
    return x, y

#@3 ------------------------------------------------------------------------------------------------
# 单卡：$ python train_gpt2.py
# DataDistributedParallel 数据分布式并行训练，8卡：
# $ torchrun --standalone --nproc_per_node=8 train_gpt2.py

#? 这三个玩意一般怎么用？accelerate封装了他们，怎么替代他们实现？
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

#@ 设置ddp，设备
# torchrun命令会设置环境变量 RANK, LOCAL_RANK, WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1  # 查看ddp有没有运行, #FIX 不太好的方式
if ddp:
  assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ['RANK'])  # 显卡编号
  ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 主机编号，不用
  ddp_world_size = int(os.environ['WORLD_SIZE'])  # 总卡的数量，一台主机即一台主机上卡的数量
  device = f'cuda:{ddp_local_rank}'
  device_type = "cuda"
  torch.cuda.set_device(device)
  master_process = ddp_rank == 0 # 主进程负责日志、保存权重等
else:
  # vanilla, non-DDP run
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  master_process = True
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
  
enc = tiktoken.get_encoding('gpt2')

#@ 设置批量大小
total_batch_size = 524288 # 2^19, ~0.5M, 单位是token的数量
B = 8 # mirco batch size， 多个微批组成一批，因为要累积梯度 (作者的B用的是16)
T = 1024
assert total_batch_size % (B * T * ddp_world_size) == 0, \
  "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)  #! 小心计算
if master_process:
  print(f"B={B}, T={T}")
  print(f"total desired batch size: {total_batch_size}")
  print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(
  B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(
  B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
# print(f"with grad accum 1 epoch = {len(train_loader.tokens) // (B * T * grad_accum_steps)} batches")
# 整个 tinyShakespeare 一共 338025 个token。 524288>338025
torch.set_float32_matmul_precision('high')  #@ 设置使用 tf32

#@ 创建模型
model = GPT(GPTConfig(vocab_size=50304))  # 50304=2^7*393，能被许多2的幂整除，性能会更好
model.to(device)
use_compile = False
if use_compile:
  model.compile()  #@ 编译模型
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])  # 模型->DDP模型
raw_model = model.module if ddp else model  # 需要在模型本体上设置优化器参数

#@ 创建优化器
# optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715  # gpt3论文，预热375Mtokens，375M/524288=715
max_steps = 10**10 // total_batch_size    # 总共训练10Btokens，10B/524288=19073
if master_process:
  print(f"max steps: {max_steps}, warmup steps: {warmup_steps}")

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

#@ 训练流程
t_st = time.time()
for step in range(max_steps):
  t0 = time.time()
  
  #@ 验证
  if step % 5 == 0:
    model.eval()
    val_loader.reset()
    with torch.no_grad():
      val_loss_accum = 0.0
      val_loss_steps = 20
      for _ in range(val_loss_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
          logits, loss = model(x, y)
        loss = loss / val_loss_steps  # 不需要loss.backward()
        val_loss_accum += loss.detach()
    if ddp:
      dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
      print(f"validation loss: {val_loss_accum.item():.4f}")

  #@ 生成文本，模型如果compile了那么这里会报错
  if step > 0 and step % 5 == 0 and not use_compile:
    model.eval()
    num_return_sequences = 4
    max_length = 32  # 生成token的最大数量
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (num, 8)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)  # 随机数生成器，防止下面的操作打乱了随机顺序
    sample_rng.manual_seed(42 + ddp_rank)
    while xgen.size(1) < max_length:
      with torch.no_grad():
        logits, loss = model(xgen) # (B, T, vocab_size)
        logits = logits[:, -1, :] # (B, vocab_size) # 取最后一个位置的 logits
        probs = F.softmax(logits, dim=-1) # (B, vocab_size) # softmax获取概率
        # 进行 top-k 采样，k=50（Hugging Face pipeline 的默认值）
        # topk_probs (4, 50)，为值 ，topk_indices (4, 50)，为索引
        topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
        # 从 top-k 概率中选择一个 token； 注意：multinomial 不要求输入的概率和为 1
        ix = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng) # (B, 1)
        # 收集对应的索引
        xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
        # 将其附加到序列中
        xgen = torch.cat((xgen, xcol), dim=1)
    for i in range(num_return_sequences): # 逐行打印
      tokens = xgen[i, :max_length].tolist()
      decoded = enc.decode(tokens) 
      print(f"rank {ddp_rank} sample {i}: {decoded}")

  #@ 训练
  optimizer.zero_grad()
  loss_accum = 0.0
  for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
      logits, loss = model(x, y) # 计算出来的是micro_batch的loss
    # cross_entropy 默认reduction=mean，损失计算公式为sum L_i/B。
    # 梯度累积的情况下使用小批量B/gas，也就是sum L_per_micro*gas/B，还需要除以一个gas。
    loss /= grad_accum_steps
    loss_accum += loss.detach()
    # 在最后一个micro_step的时候才运行backward计算；减少backward次数。因为backward会卡间通信，很慢；非标准方式
    if ddp:
      model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)  
    loss.backward()  #@ require_backward_grad_sync为true的时候会，会在多卡间同步梯度，默认做平均
  if ddp:  # 累加平均每个gpu上的loss
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
  ##$ 使用等比缩放，控制全梯度L2范数不超过1.0（hack trick）；返回原始全梯度的L2范数
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 控制梯度最重要的目的是防止梯度爆炸
  lr = get_lr(step)  # 获取学习率
  for param_group in optimizer.param_groups:  # 需手动更新每一个参数组的学习率
    param_group['lr'] = lr 
  optimizer.step()
  # cpu给gpu发指令，很快就会运行到下面，但此时gpu还没有执行完。于是手动等待同步
  if device_type == "cuda":
    for i in range(torch.cuda.device_count()):
      torch.cuda.synchronize(i)
    # torch.cuda.synchronize(device) 
  t1 = time.time()
  dt = t1 - t0
  tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size # 524288
  tokens_per_sec = tokens_processed / dt
  if master_process:
    print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} |", \
        f"dt: {dt*1000:.2f}ms | tok/sec {tokens_per_sec:.2f}")
  
t_ed = time.time()
print(f"total training time: {t_ed - t_st:.2f}s")
print(f"tok/sec: {50*train_loader.B*train_loader.T / (t_ed - t_st):.2f}")

if ddp:
  destroy_process_group()
import sys; sys.exit(0)

#$ ------------------------------------------------------------------------------------------------
# prefix tokens
model = GPT.from_pretrained('gpt2')
model.to(device)  # 不需要model=model.to(device)
model.eval()
num_return_sequences = 5
max_length = 30



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
