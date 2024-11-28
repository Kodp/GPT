from dataclasses import dataclass

# 配置文件
@dataclass
class GPTConfig:
  block_size: int = 1024   # 语言模型的输入序列长度
  vocab_size: int = 50257  # 50000 BPE merge + 256 bytes tokens(基础符号) + 1 <|endoftext|> token
  n_layer   : int = 12     # 层数
  n_head    : int = 12     # 多头注意力的头数
  n_embd    : int = 768    # Embedding维度


config1 = GPTConfig()
config2 = GPTConfig()
# 自动重写了__eq__方法，比较两个对象是否相等
print(config1 == config2)  # 输出：True

# 自动重写了__repr__方法，打印对象时输出的字符串
print(config1) # 输出：GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768)

# 自定义部分参数
config3 = GPTConfig(n_layer=2400, n_head=16)
print(config3) # 输出：GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=768)

# 访问
print(config3.n_layer) 
# 修改
config3.n_layer = 12
print(config3.n_layer)