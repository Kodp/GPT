# 美化错误输出 ---------------------------------------------------------------------------------------
import inspect
from librosa import example
from rich.traceback import install
install()
#---------------------------------------------------------------------------------------------------

# timer 装饰器 --------------------------------------------------------------------------------------
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
# --------------------------------------------------------------------------------------------------

from dataclasses import dataclass
from torch import logit, nn
from torch.nn import functional as F
import torch
import math
import os
import time
from hellaswag import render_example, iterate_examples # HellaSwag

# 配置文件
@dataclass
class GPTConfig:
  block_size: int = 1024   # 语言模型的输入序列长度
  vocab_size: int = 50257  # 50000 BPE merge + 256 bytes tokens(基础符号) + 1 <|endoftext|> token
  n_layer   : int = 12     # 层数
  n_head    : int = 12     # 多头注意力的头数
  n_embd    : int = 768    # Embedding维度


#@ 多头自注意力层
class CausalSelfAttention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    # 保证多头输出cat后维度为n_embd（当然也可以不对齐，proj改一下输入维度=头数*head_size)
    assert config.n_embd % config.n_head == 0  
    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  #(C,3C) W_q, W_k, W_v, 水平拼接
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1
    self.n_head = config.n_head
    self.n_embd = config.n_embd

    # register_buffer 注册一个缓冲区张量——不参与梯度计算，也不更新。
    # torch.tril 生成下三角矩阵
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)) \
      .view(1, 1, config.block_size, config.block_size))   # (1, 1, block_size, block_size)
    """
    bias = tensor([[[[1., 0., 0., 0.],
                     [1., 1., 0., 0.],
                     [1., 1., 1., 0.],
                     [1., 1., 1., 1.]]]])
    """
      
  def forward(self, x):
    """前向传播
    nh = number of heads, hs = head size, T = Time-step/sequence length, C=hidden size
    """
    B, T, C = x.size()
    qkv:torch.Tensor  = self.c_attn(x) # (B,T,C)@(C,3C)->(B,T,3C)
    q, k, v = qkv.split(self.n_embd, dim=2) # 3 x (B,T,C)
    #todo 需要自己实现一个tensor类、支持这种功能，才能真正理解数据是怎么变换的。
    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))) # (B, nh, T, T)
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')) # 防止偷看未来的token
    att = F.softmax(att, dim=-1) # (B, nh, T, T)
    y   = att @ v # (B,nh,T,T)@(B,nh,T,hs)->(B,nh,T,hs)
    y   = y.transpose(1, 2).contiguous().view(B, T, C)  
    
    y = self.c_proj(y)
    return y

class MLP(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
    self.gelu = nn.GELU(approximate='tanh')
    self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    self.c_proj.NANOGPT_SCALE_INIT = 1
    
  def forward(self, x):
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

# Pre-Norm Transformer
class Block(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    self.ln_1 = nn.LayerNorm(config.n_embd)
    self.attn = CausalSelfAttention(config)
    self.ln_2 = nn.LayerNorm(config.n_embd)
    self.mlp = MLP(config)
  
  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


class GPT(nn.Module): # 继承基类，可以自动反向传播
  def __init__(self, config: GPTConfig):
    super().__init__() # 调用父类构造函数，必须执行
    self.config = config
    
    self.transformer = nn.ModuleDict(dict( # 无序存储一些用到的module
      wte=nn.Embedding(config.vocab_size, config.n_embd), # token嵌入查找表,长度为词表长度
      wpe=nn.Embedding(config.block_size, config.n_embd), # pos嵌入查找表, 长度为输入序列的长度
      h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  
      ln_f=nn.LayerNorm(config.n_embd),
    ))
    # 语言模型的最后一层通常是一个线性层加上 softmax，用于预测下一个单词的概率分布。
    # 去掉偏置项可以简化模型，减少参数量，而且对性能影响不大。
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    # weight sharing scheme 共享softmax之前的线性层和embedding层权重
    self.transformer.wte.weight = self.lm_head.weight  # 原来的wte.weight会变成孤立状态，从而被回收
    
    self.apply(self._init_weights)
    
  # 对照gpt2原始tensorflow代码的初始化
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      std = 0.02
      if hasattr(module, 'NANOGPT_SCALE_INIT'):  
        std *= (2 * self.config.n_layer) ** -0.5 # 2 来自两次残差连接
      torch.nn.init.normal_(module.weight, mean=0.0, std=std)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
  def forward(self, idx: torch.Tensor):
    """idx.shape=(B,T)"""
    B, T = idx.size()
    assert T <= self.config.block_size, \
      f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
    # forward the token and pos embedding
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    pos_emb = self.transformer.wpe(pos) # (T, n_embd)
    tok_emb = self.transformer.wte(idx) # (B, T, n_embd) 按 idx[i][j] 的值，取Embed表的对应行-(,n_embd)
    x = tok_emb + pos_emb # 隐式广播，在Batch维度上，pos_emb(T, n_embd)->(B, T, n_embd), 
    # forward the blocks of transformer
    for block in self.transformer.h:
      x = block(x)
    # forward the final layernorm and the classisfier
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)   # (B, T, vocab_size)
    
    #@ 最后结果(B, T, vocab_size)：对于T个过去的token，预测下一个token在vocab中的概率分布，size=vocab_size。
    # 通过argmax(logits, dim=-1)就可以得到预测的下一个token
    return logits
    
  @classmethod # 类方法，无须实例化。第一个cls表示类本身。
  def from_pretrained(cls, model_type):
    """从huggingface加载预训练gpt2模型"""
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    from transformers import GPT2LMHeadModel  # 作用域在函数内
    print("loading weights from pretrained gpt: %s" % model_type)
    
    config_args = {
      'gpt2'       : dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
      'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
      'gpt2-large' : dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
      'gpt2-xl'    : dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
    }[model_type]
    
    config_args['vocab_size'] = 50257  # 对于gpt模型都是50257
    config_args['block_size'] = 1024
    
    config = GPTConfig(**config_args) # GPTConfig: dataclass
    model: nn.Module = GPT(config)    # 创建一个GPT类实例（执行__init__）
    sd = model.state_dict()           #@ 我们模型的参数(weight,bias)和buffer
    sd_keys = sd.keys()  #$ sd=state_dict
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # 去掉mask/buffer, 没有参数
    
    #@ 加载HF/transfomers的GPT2模型, 并把参数拷贝到我们的模型
    model_hf: nn.Module = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
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
    根据参数维度将参数分为两组，分别应用不同的weight decay。控制台输出需要优化的参数大小。
    创建并返回AdamW优化器，若设备为CUDA且支持，使用fused版本的AdamW。
    """
    # 获取所有需要梯度更新的参数
    param_dict = {pn: p for pn, p in self.named_parameters()}  # .named_parameters() 返回参数名+参数本身
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # 挑选需要梯度的
    # 创建需要decay的参数组：所有>=2维的张量的list；低于2维则nodecay
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    ## 设置字典
    optim_groups = [
      {'params':decay_params, 'weight_decay':weight_decay},
      {'params':nodecay_params, 'weight_decay': 0.0},
    ]
    # 计算两组需要优化的参数量
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    # 主进程打印参数量
    if master_process:   #! 下面ddp里的
      # {int:,} `,`表示千分位格式化。"1000000" -> "1,000,000"
      print(f"num decayed parameter tensors: {len(decay_params)}, \
            with {num_decay_params:,} parameters")
      print(f"num non-decayed parameter tensors: {len(nodecay_params)}, \
            with {num_nodecay_params:,} parameters")
    # 检查能否使用fused版本的AdamW优化器并尝试使用
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    if master_process:
      print(f"using fused AdamW: {use_fused}")
    optimizer = torch.optim.AdamW(
      optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer
    
# --------------------------------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename) -> torch.Tensor:
  npt = np.load(filename)
  npt = npt.astype(np.int32)  # 视频之后添加
  ptt = torch.tensor(npt, dtype=torch.long)
  return ptt

class DataLoaderLite:
  """
  数据加载器类，用于按分片加载训练或验证数据。支持多进程并行加载，返回指定批次大小和时间步长的数据。
  每次加载一个批次，自动切换到下一个数据分片以避免超出范围。
  """
  def __init__(self, B, T, process_rank, num_processes, split):
    #
    # 初始化
    self.B = B # 批次大小
    self.T = T # 时间步长
    self.process_rank = process_rank    # 当前进程的编号
    self.num_processes = num_processes  # 总进程数
    assert split in {'train', 'val'} 
    # 获取数据分片文件名
    data_root = "edu_fineweb10B"
    shards = os.listdir(data_root)
    shards = [s for s in shards if split in s]
    shards = sorted(shards)
    shards = [os.path.join(data_root, s) for s in shards]
    self.shards = shards
    assert len(shards) > 0, f"no shards found for split {split}"
    if master_process:
      print(f"found {len(shards)} shards for split {split}")
    self.reset()
    
  def reset(self):
    self.current_shard = 0  # 当前分片的索引
    self.tokens = load_tokens(self.shards[self.current_shard])  #@ 加载当前分片的数据
    self.current_position = self.B * self.T * self.process_rank # 当前进程在数据中的起始位置
    # 记录取下一个batch的起始位置
  
  def next_batch(self):
    B, T = self.B, self.T
    buf = self.tokens[self.current_position : self.current_position+B*T+1]
    x = (buf[:-1]).view(B, T)  # inputs
    y = (buf[1:]).view(B, T)   # targets
    # 更新当前进程的位置
    self.current_position += B * T * self.num_processes
    # 如果下一批次的末端超过当前分片数据范围，则加载下一个分片 #? 什么意思？
    if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
      self.current_shard = (self.current_shard + 1) % len(self.shards)
      self.tokens = load_tokens(self.shards[self.current_shard]) # 加载新分片数据
      self.current_position = B * T * self.process_rank          #? 重置进程位置
    return x, y
    
# --------------------------------------------------------------------------------------------------
# 
def get_most_likely_row(tokens:torch.Tensor, mask:torch.Tensor, logits:torch.Tensor):
  """辅助 HellaSwag 数据集评估函数。
  计算并返回最可能的补全对应的行索引。
  首先计算每个位置的损失，然后在补全区域（mask == 1）内求平均损失，最后返回具有最低损失的行索引作为最可能的补全结果。
  """
  #@ B: Batch size, T: sequence length/ Time-step, C: vocab_size
  #? 一次计算4个选项，但是长度怎么对齐的？合理吗，直接用目标答案的token来计算损失？万一模型前几个词汇预测没有生成答案呢
  # 去掉最后一个token的logits 
  shift_logits = (logits[..., :-1, :]).contiguous() # (B,T-1,C)
  # 去掉第一个token
  shift_tokens = (tokens[..., 1:]).contiguous()    #  (B,T-1)
  # 展平，方便计算交叉熵
  flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) #  (B*(T-1),C) 预测的logits
  flat_shift_tokens = shift_tokens.view(-1)  #  (B*(T-1),) 实际的token
  # 计算每个位置的交叉熵损失；reduction='none'表示返回每个位置的损失值，而非平均或求和。
  shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
  
  shift_losses = shift_losses.view(tokens.size(0), -1)
  shift_mask = (mask[..., 1:]).contiguous()
  
  # 将损失和掩码相乘，掩码为1的地方保留损失，掩码为0的地方将损失设为0。
  masked_shift_losses = shift_losses * shift_mask
  # 对于每一行，计算所有非零掩码的损失和，并除以掩码中1的数量，得到平均损失。
  sum_loss = masked_shift_losses.sum(dim=1) 
  avg_loss = sum_loss / shift_mask.sum(dim=1)
  
  # now we have a loss for each of the 4 completions
  # the one with the lowest loss should be the most likely
  # 挑出最小损失的索引，argmin()返回最小值的索引，item()将其转为一个标量值。
  pred_norm = avg_loss.argmin().item() 
  
  return pred_norm


# -------------------------------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

#@ 分布式训练设置
# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE

ddp = int(os.environ.get('RANK', -1)) != -1  # ddp 有没有在跑
if ddp:
  assert torch.cuda.is_available(), "for now i thnk we need CUDA for DDP"
  init_process_group(backend='nccl')
  ddp_rank = int(os.environ['RANK'])
  ddp_local_rank = int(os.environ['LOCAL_RANK'])
  ddp_world_size = int(os.environ['WORLD_SIZE'])
  device = f'cuda:{ddp_local_rank}'
  torch.cuda.set_device(device)
  master_process = ddp_rank == 0 # 主进程负责logging，checkpointing，etc.
else:
  # 单卡训练
  ddp_rank = 0
  ddp_local_rank = 0
  ddp_world_size = 1
  master_process = True
  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"  # Apple
  print(f"using device: {device}")

# 视频之后添加，pytorch里严格区分device和device_type
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")
total_batch_size = 524288  # 2**19, ~0.5M，是token的数量
B = 64   # micro batch size
T = 1024 # context length 上下文长度
assert total_batch_size % (B * T * ddp_world_size) == 0, \
  "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
  print(f"total desired batch size: {total_batch_size}")
  print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

#@ 设置loader
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, 
                              split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, 
                            split="val")

torch.set_float32_matmul_precision('high')  #? 干嘛？

# 创建模型
model = GPT(GPTConfig(vocab_size=50304)) 
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False # torch.compile #?
if use_compile:
  model = torch.compile(model)
if ddp:
  model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  #? always contains the "raw" unwrapped model?

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 在10Btoken+batchsize=0.5M的情况下，19073step大约是1epoch #？

def get_lr(it):
  """学习率调整"""
  # 1. 如果当前的训练步数（it）小于预热步数（warmup_steps），则进行线性预热
  if it < warmup_steps:
    return max_lr + (it + 1) / warmup_steps
  # 2. 如果当前训练步数大于最大步数（max_steps），则直接返回最小学习率（min_lr）
  if it > max_steps:
    return min_lr
  # 3. 如果当前训练步数处于预热和最大步数之间，则使用余弦衰减的方式计算学习率
  decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
  assert 0 <= decay_ratio <= 1
  # 该系数会随着训练的进展从1逐渐衰减到0
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  # 系数从1衰减到0，因此最终学习率将从最大值衰减到最小值
  return min_lr + coeff * (max_lr - min_lr)

# 设置优化器
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, 
                                           device_type=device_type)

# logdir, 用于存储日志和模型
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:  # 只是为了清空文件
  pass

#@ 开始训练
#! 需细细的处理分析，这里目前只是敲了一遍
for step in range(max_steps):
  t0 = time.time()
  last_step = (step == max_steps - 1)
  
  #@ 定期评估验证损失
  if step % 250 == 0 or last_step:
    model.eval()
    val_loader.reset()
    with torch.no_grad():
      val_loss_accum = 0.0
      val_loss_steps = 20 # 每次计算损失时的步数
      for _ in range(val_loss_steps):
        x, y = val_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16): # 自动混合精度
          logits, loss = model(x, y)
        loss = loss / val_loss_steps  # 平均损失
        val_loss_accum += loss.detach()
    ## 如果是分布式训练，执行所有进程之间的平均
    if ddp:
      dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
    ##@ 主进程输出log，保存模型
    if master_process:
      print(f"validation loss: {val_loss_accum.item():.4f}")
      with open(log_file, "a") as f:
        f.write(f"{step} val {val_loss_accum.item():.4f}\n")
      ### 保存checkpoint
      if step > 0 and (step % 5000 == 0 or last_step):
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = {
          'model': raw_model.state_dict(),
          'config': raw_model.config,
          'step': step,
          'val_loss': val_loss_accum.item(),
        }
        torch.save(checkpoint, checkpoint_path)  #? torch保存了一个python字典？save功能和特性？
  
  #@ 定期评估HellaSwag任务
  if (step % 250 == 0 or last_step) and (not use_compile):
    num_correct_norm = 0 # 正确的预测数
    num_total = 0        # 总的预测数
    for i, example in enumerate(iterate_examples("val")):
      if i % ddp_world_size != ddp_rank:
        continue
      _, tokens, mask, label = render_example(example)  # 生成-模型需要的输入
      tokens = tokens.to(device)
      mask = mask.to(device)
      with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16): #? 具体做了什么？
          logits, loss = model(tokens)  # 获取模型输出
        pred_norm = get_most_likely_row(tokens, mask, logits)
      num_total += 1
      num_correct_norm += int(pred_norm == label) # 统计正确的预测
    ## 如果是分布式训练，汇总所有进程的统计数据
    if ddp:
      num_total = torch.tensor(num_total, dtype=torch.long, device=device)
      num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
      #* 因为还没做csapp倒数第二个lab，所以不懂并行是怎么弄的 
      dist.all_reduce(num_total, op=dist.ReduceOp.SUM) 
      dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
      num_total = num_total.item()
      num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    if master_process:
      print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
      with open(log_file, "a") as f:
        f.write(f"{step} hella {acc_norm:.4f}\n")
  
  #@ 定期生成模型输出（排除第一次）
  if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile): #? compile怎么就不行了
    model.eval()
    num_return_sequences = 4  # 生成序列数
    max_length = 32           # 生成序列长度
    tokens = enc.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (num_r_s, t)
    xgen = tokens.to(device)
    sample_rng = torch.Generator(device=device)  # 随机数生成器
    sample_rng.manual_seed(42 + ddp_rank)
    ## 生成的tokens长度小于最大长度就持续生成
    while xgen.size(1) < max_length:
      with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
          logits, loss = model(xgen)
      logits = logits[:, -1, :]  # 对最后一个token的预测分布
      probs = F.softmax(logits, dim=-1)
      topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)   #? 为什么不直接采样
      ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # 从topk中采样一个
      xcol = torch.gather(topk_indices, -1, ix)  #? 获取选择的token
      xgen = torch.cat((xgen, xcol), dim=1)  # 添加新token到序列
    ## 打印生成的文本
    for i in range(num_return_sequences):
      tokens = xgen[i, :max_length].tolist()  #? :max_length多此一举？
      decoded = enc.decode(tokens)
      print(f"rank {ddp_rank} sample {i}: {decoded}")
    
  #@ 优化
  model.train()
  optimizer.zero_grad()
  loss_accum = 0.0
  for micro_step in range(grad_accum_steps):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    if ddp: # 如果是分布式训练，设置同步梯度
      model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) #?
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
      logits, loss = model(x, y)  # 前向传播
    loss = loss / grad_accum_steps
    loss_accum += loss.detach()
    loss.backward()
  if ddp:  # 如果是分布式训练，汇总梯度
    dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
  norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  #? 梯度裁剪
  lr = get_lr(step)  # 获取当前学习率
  ## 更新优化器的学习率 怎么tm在手动做啊
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  optimizer.step()  #@ 优化器更新
  if device_type == "cuda":  # 确保同步
    torch.cuda.synchronize()
    
  t1 = time.time()
  dt = t1 - t0
  tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size # 总token数
  tokens_per_sec = tokens_processed / dt  # 每秒处理的tokens数
  #@ 日志
  if master_process:
    print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f}" + \
      f" | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    with open(log_file, "a") as f:
      f.write(f"{step} train {loss_accum.item():.6f}\n")
      
# 结束分布式训练时销毁进程组
if ddp:
  destroy_process_group()
  
  
      
      
      

    
    

      
      
      
           
      




