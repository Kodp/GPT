# 美化错误输出 ---------------------------------------------------------------------------------------
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
  
  
# --------------------------------------------------------------------------------------------------
num_return_sequences = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()  # 从上面定义来看，没有任何层是training不同于testing的；也许torch会做些聪明的事情:)
model.to('cuda')
print("model loaded to cuda")

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")  # 最后加空格，输出会混乱，原因主要出在token上
tokens = torch.tensor(tokens, dtype=torch.long)  # long=int64; (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to('cuda')

# 生成，(B, T) ==> (5, 8)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
  # forward the model to get logits
  with torch.no_grad():
    logits = model(x) # (B, T, vocab_size)
    logits = logits[:, -1, :] # (B, vocab_size)
    probs = F.softmax(logits, dim=-1)  
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    ix = torch.multinomial(topk_probs, 1)
    xcol = torch.gather(topk_indices, -1, ix)
    x = torch.cat((x, xcol), dim=1)

# 输出生成文本
for i in range(num_return_sequences):
  tokens = x[i, :max_length].tolist()
  decoded = enc.decode(tokens)
  print(">", decoded)
    
    
