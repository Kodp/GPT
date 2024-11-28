import torch
import torch.nn as nn
from torch.nn import functional as F

# #### 1. 设置超参数
batch_size = 32
context_size = 8  # Bigram model uses a context size of 1
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
seed = 42
torch.manual_seed(seed)

# #### 2.读取数据集
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# #### 3.建立 tokenizer (`decode` `encode` 函数)
# 字符级别分词
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(indices):
    return ''.join([itos[i] for i in indices])

# #### 4. 分割训练集，验证集
# `90%` 训练
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# #### 5. 获取一个 batch 的函数
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    return x.to(device), y.to(device)

# #### 6. 计算 loss 函数（train and test）
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X_batch, Y_batch = get_batch(split)
            logits = model(X_batch)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), Y_batch.view(B*T))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# > 模型的 forward 可以不算 loss 吗？
# 是的，模型的 forward 方法可以只计算输出，不计算 loss。loss 通常在模型外部计算。

# #### 7. 模型
class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super(BigramModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        logits = self.embedding(idx)  # (B, T, C)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)  # (B, T, C)
            logits = logits[:, -1, :]  # (B, C)
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_idx), dim=1)  # (B, T+1)
        return idx

# #### 8. 主函数
# 初始化模型、优化器
model = BigramModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 循环 `max_iters`
for iter in range(max_iters):
    # 评估并打印 loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 获取 batch，训练
    X_batch, Y_batch = get_batch('train')
    logits = model(X_batch)  # (B, T, C)
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B*T, C), Y_batch.view(B*T))

    # 优化器清空梯度，loss 反向传播，优化器 step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# #### 9.生成
context = torch.zeros((1, 1), dtype=torch.long).to(device)  # 用零初始化
generated = model.generate(context, max_new_tokens=1000)
print(decode(generated[0].tolist()))
