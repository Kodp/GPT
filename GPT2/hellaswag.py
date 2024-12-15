"""
在 Python 环境中可下载并评测 HellaSwag：  
https://github.com/rowanz/hellaswag

HellaSwag 验证集共计 10,042 个样本。

HellaSwag JSON 示例：  
{
  "ind": 24,                                // 编号  
  "activity_label": "Roof shingle removal", // 对应的 ActivityNet 或 WikiHow 标签  
  "ctx_a": "A man is sitting on a roof.",   // 上下文的前面部分放入 ctx_a。
  "ctx_b": "he",                // 上下文以不完整名词短语结尾如在 ActivityNet 中，该不完整短语放入 ctx_b
  "ctx": "A man is sitting on a roof. he",  // 完整上下文，ctx = ctx_a + " " + ctx_b。
  "split":   "val",                         // 数据集划分类型，可为 train、val 或 test。
  "split_type": "indomain",                 // 数据集分布是否和训练集一致。一致indomain，不一致zero-shot
  "label": 3, 
  "endings": [                              // 4个结尾候选项，正确的选项由label给出
    "is using wrap to wrap a pair of skis.", 
    "is ripping level tiles off.", 
    "is holding a rubik's cube.",  
    "starts pulling up roofing on a roof."
    ], 
  "source_id": "activitynet~v_-JhWjGDPHMY"   // 来源的视频或 WikiHow 条目标识
}

gpt2 (124M) ：  
- eleuther harness reports (multiple choice style)：acc:28.92%，acc_norm:31.14%  
- 本脚本（补全模式）：在10042条验证样本上，acc:0.2859，acc_norm:0.2955

gpt2-xl (1558M)：  
- eleuther harness（多选模式）：acc:40.04%，acc_norm:50.89%  

HellaSwag 的验证集含10042个样本。
"""
from rich.traceback import install; install()
import os, json, requests, tiktoken
from tqdm import tqdm
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from pathlib import Path
#%------------------------------------------------------------------------------------------------

DATA_CACHE_DIR = str(Path(__file__).parent / "hellaswag") 

def download_file(url: str, filename:str, chunk_size=1024):
  """下载指定url的文件"""
  resp = requests.get(url, stream=True)
  total = int(resp.headers.get("content-length", 0))
  with open(filename, "wb") as file, tqdm(
    desc=filename,
    total=total,
    unit="iB",
    unit_scale=True,
    unit_divisor=1024,
  ) as bar:  # tqdm: 1.0 KiB = 1024B
    for data in resp.iter_content(chunk_size=chunk_size):
      size = file.write(data)
      bar.update(size)
  

hellaswags = {
  "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
  "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
  "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
  os.makedirs(DATA_CACHE_DIR, exist_ok=True)
  data_url = hellaswags[split]
  data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl") # 每行包含一个完整的JSON对象
  # if not Path(data_filename).exists():
  if not os.path.exists(data_filename):
    print(f"Downloading {data_url} to {data_filename}...")
    download_file(data_url, data_filename)
    
def render_example(example:dict):
  """
  将字典example转换为三个torch.tensor:
  - tokens: 上下文+补全,候选项4个，size(4,T)
  - mask: 候选项位置全为1，之后在1处计算可能性, size(4,T)
  - label: 正确选项索引，希望模型在这个选项上有最大可能性 int
  """
  ctx = example['ctx']
  label = example['label']
  endings = example['endings']
  
  data = {
    "label": label,
    "ctx_tokens": None,
    "ending_tokens": [],
  }
  # 汇聚所有的token
  ctx_tokens = enc.encode(ctx)
  data["ctx_tokens"] = ctx_tokens
  tok_rows = []
  mask_rows = []
  for end in endings:
    # 前置空格性能会好一些。因为输入ctx预测后面end，ctx的末尾和end的开头都没有空格，直接拼接在一起变成
    # A man is sitting on a roof. heis ripping level tiles off. heis连在一起变成标答，是不对的。
    # 而且模型更可能预测" is" 而不是没有空格的"is"。
    end_tokens = enc.encode(" " + end) 
    data["ending_tokens"].append(end_tokens)
    tok_rows.append(ctx_tokens + end_tokens)
    mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
    
  # 由于不同回答长度不同，统一为最大值，这样才能变成一个二维tensor；如果每一行长度不一样处理上会更复杂
  max_len = max(len(row) for row in tok_rows)
  tokens = torch.zeros((4, max_len), dtype=torch.long)
  mask = torch.zeros((4, max_len), dtype=torch.long)
  ## 设置token和mask
  for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
    tokens[i, :len(tok_row)] = torch.tensor(tok_row)
    mask[i, :len(mask_row)] = torch.tensor(mask_row)
  
  return data, tokens, mask, label

# 生成器函数
def iterate_examples(split):
  # 10042 个样本
  download(split)
  with open(Path(DATA_CACHE_DIR) / f"hellaswag_{split}.jsonl", "r") as f:
    for line in f:
      example = json.loads(line)
      yield example  # 函数会变为 generator，每次执行到 `yield x` 停止、返回 `x`

@torch.no_grad()
def evaluate(model_type, device):
  torch.set_float32_matmul_precision('high') # use tf32
  model = GPT2LMHeadModel.from_pretrained(model_type)
  model.to(device)
  
  num_correct_norm = 0
  num_correct = 0
  num_total = 0
  
  for example in iterate_examples("val"):
    data, tokens, mask, label = render_example(example)
    tokens = tokens.to(device)  # (4,T)
    mask = mask.to(device)
    #hint 由于每个选项共享同样的前导序列，所以模型预测的logits每一个(T,V)张量都是一样的
    logits = model(tokens).logits  # (4,T,V) # V=vocab_size=50257，T=20
    shift_logits = (logits[..., :-1, :]).contiguous() # (4,T-1,V)
    shift_tokens = (tokens[..., 1:]).contiguous() # (4,T-1)
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1)) # (4*T-4,v)
    flat_shift_tokens = shift_tokens.view(-1) # (4*T-4,)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none') # (4*T-4,)
    shift_losses = shift_losses.view(tokens.size(0), -1) # (4,T-1)
    
    shift_mask = (mask[..., 1:]).contiguous()  # (4,T-1)
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1) # (4,) # 所有需要预测token的位置计算loss的总和
    avg_loss = sum_loss / shift_mask.sum(dim=1) # (4,) # 总和除以需要预测的token数量，平均每个位置的loss
    
    pred = sum_loss.argmin().item()
    pred_norm = avg_loss.argmin().item()
    
    num_total += 1
    # 总损失最小化的策略得到的正确预测数量
    num_correct += int(pred == label)  
    # 平均损失最小化的策略得到的正确预测数量，对异常值的敏感度较低，可能更能准确地反映模型的性能
    num_correct_norm += int(pred_norm == label)  
    
    if num_total % 100 == 0 or num_total == 10042:  # 一共10042个
      print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

    if num_total < 10:
      print("-" * 72)
      print(f"Context:\n {example['ctx']}")
      print(f"Endings:")
      for i, end in enumerate(example["endings"]):
        print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
      print(f"Predicted: {sample_masked(model, example['ctx'], logits, mask)}")
      print(f"predicted: {pred_norm}, actual: {label}")
      
def sample_masked(model, context:str, logits:torch.Tensor, mask:torch.Tensor):
  """
  返回掩码为1的部分模型预测的token对应的文本, top采样
  logits:(B,T,V)
  mask:  (B,T)
  """
  B, T = mask.shape
  # 由于每个选项共享同样的前导序列，所以模型预测的logits每一个(T,V)张量都是一样的, 取第一行计算即可
  tokens = enc.encode(context)
  tokens = torch.tensor(tokens, dtype=torch.long, device=mask.device) # (T,)
  sample_rng = torch.Generator(device=mask.device); sample_rng.manual_seed(42)
  
  row = torch.argmax(mask.sum(dim=1))  # 最多1的行。
  # 因为前导一样的，rngseed一样的，不同选项的生成只有长度的区别。所以，选最长的行的mask
  # 我也不希望对不同的选项用不同的seed，否则对每个选项
  # 生成的结果不同，对于眼观其loss只有干扰
  
  travel = False
  for j in range(T):
    if mask[row][j].item() == 0:
      if travel == False: continue
      else: break
    else:
      travel = True
    logits = model(tokens).logits  # (T+,V) # V=vocab_size=50257，T=20
    # [ -33.5478,  -32.7478,  -35.4299,  ..., -40.1671, -33.1929] 没有过 softmax之前，logits大概是这样
    probs = F.softmax(logits[-1], dim=-1) #Implicit dimension choice for softmax has been deprecated.
    topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
    topk_idx = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)  # 采样的得topk索引
    token = torch.gather(topk_indices, dim=-1, index=topk_idx) # topk索引->原始索引（token）
    tokens = torch.cat((tokens, token), dim=-1)
  text = enc.decode(tokens.tolist())  
  
  return text[len(context):]
  

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
  parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use") 
  args = parser.parse_args()
  evaluate(args.model_type, args.device)
  


  
  
    
  