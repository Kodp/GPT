import torch
import torch.nn as nn
import torch.optim as optim

# 定义嵌入层模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.wpe = nn.Embedding(2, 1)  # E_pe: (2, 1)
        self.wte = nn.Embedding(3, 1)  # E_te: (3, 1)

    def forward(self, pos, idx):
        pos_emb = self.wpe(pos)  # (2, 1)
        tok_emb = self.wte(idx)  # (2, 1)
        x = tok_emb + pos_emb    # (2, 1)
        return x

# 初始化模型
model = SimpleModel()

# 手动设置嵌入矩阵的初始值
with torch.no_grad():
    model.wpe.weight = nn.Parameter(torch.tensor([[1.0], [2.0]]))
    model.wte.weight = nn.Parameter(torch.tensor([[3.0], [4.0], [5.0]]))

# 定义输入索引
pos = torch.tensor([0, 1], dtype=torch.long)  # [0, 1]
idx = torch.tensor([1, 2], dtype=torch.long)  # [1, 2]

# 定义目标输出
target = torch.tensor([[4.9], [6.8]], dtype=torch.float)

# 定义损失函数
criterion = nn.MSELoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 前向传播
output = model(pos, idx)  # [[5.0], [7.0]]
loss = criterion(output, target)
print(f"初始损失: {loss.item():.4f}")  # 初始损失: 1.0

# 反向传播
loss.backward()

# 查看梯度
print("wpe 的梯度:\n", model.wpe.weight.grad)
print("wte 的梯度:\n", model.wte.weight.grad)

# 更新参数
optimizer.step()

# 清零梯度
optimizer.zero_grad()

# 查看更新后的嵌入矩阵
print("更新后的 E_pe:\n", model.wpe.weight)
print("更新后的 E_te:\n", model.wte.weight)

# 再次前向传播，计算新的损失
output = model(pos, idx)
loss = criterion(output, target)
print(f"更新后的损失: {loss.item():.4f}")