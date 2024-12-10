"""
FineWeb-Edu 数据集
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
下载并令牌化数据，分片存储到磁盘，路径为 ./edu_fineweb10B。
运行：
$ python fineweb.py

- 使用函数 `datasets.load_dataset` 下载数据集
- 写好分词处理函数，之后多进程使用
	- 分词器使用 `tiktoken.get_encoding("model_name")`
	- 使用 `np.uint16` 格式储存
- 用 mp（multiprocessing）开启多进程
	- 预分配缓冲区，缓冲区填满了就保存到磁盘
	- imap 分配，执行任务，返回一个个数据 `x`
		- 如果填了数据后缓冲区未满，则填数据 `x`；
		- 否则把填满缓冲，保存到磁盘，然后把 `x` 剩下的部分填入新缓冲区。（要保证缓冲区大小>数据大小）
	- 保存缓冲区中剩下的数据
"""

import os
from pathlib import Path
from datasets import load_dataset  # pip install datasets
import tiktoken
import numpy as np
import multiprocessing as mp
from tqdm import tqdm  # pip install tqdm
# os.environ["HF_DATASETS_OFFLINE"] = "1" # 只在当前进程中生效，强制使用本地数据

shard_save_dir = "edu_fineweb10B1"  # 保存路径文件名
# # shard_size 要足够大，要大于每一个文章的token数量，否则buffer里存不下、出错 
shard_size = int(1e8) # 100M tokens/shard， 总共100个分片 
shard_prefix = "edufineweb"  # 保存的数据分片的文件名前缀
SHARD_DIR = Path(__file__).parent / shard_save_dir # __file__ 绝对路径， / 重载
os.makedirs(SHARD_DIR, exist_ok=True)

data = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")  # 加载数据集，含10Btokens
#debug: 观察到主进程中内存持续增长直到和数据集大小同数量级的问题。
# 原因是load_dataset之后的data对象（Hugging Face Dataset）在迭代访问中将数据逐步读入内存，
# 通过Arrow内存映射还是某种内部缓存机制，当for tokens in pool.imap(...):循环遍历整个数据集时，主进程实际上
# 已经对全部数据进行了一次读取，从而在内存使用上体现出与数据大小一致的增长。
#fix: 由于仅需要顺序遍历一次数据集，而且不需要随机访问，所以可在load_dataset里添加streaming=True参数。
# 注意使用streaming=True后，无法直接对数据集进行切片或获取长度(len)。
# 而且还需要设置.batch、prefetch、num_proc才能快。只有内存真的不够再用。

enc = tiktoken.get_encoding("gpt2")  # 编码器
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc_dict:dict):
  """处理文本，开头添加[eot]"""
  tokens = [eot]
  tokens.extend(enc.encode_ordinary(doc_dict["text"])) # encode_oridinary 编码时忽略特殊词元
  tokens_np = np.array(tokens)
  assert (0 <= tokens_np).all() and (tokens_np < 2 ** 16).all(), "Too large for uint16"
  tokens_np_uint16 = tokens_np.astype(np.uint16)  # 词汇表大小50257，uint16足够，相比int32省下一半空间
  return tokens_np_uint16


def write_datafile(filename, tokens_np):
  """为什么抽象？
  因为以后如果需要在存储前后执行一些额外操作（如记录保存时间、打印提示信息或者进行数据校验），
  只需要修改这个函数。"""
  np.save(filename, tokens_np)

if __name__ == '__main__':
  nprocs = os.cpu_count() // 2  # 使用一半的CPU核心数已经足够快？
  with mp.Pool(processes=nprocs) as pool:  # 开启进程池，nprocs个进程
    shard_index = 0
    buffer = np.empty((shard_size,), dtype=np.uint16)  # buffer size = shard size
    current_size = 0
    progress_bar = None
    # imap，每个进程处理chunksize次func，参数为data[i]；返回结果的顺序与输入 data 的顺序一致
    for tokens in pool.imap(func=tokenize, iterable=data, chunksize=32):  
      if current_size + len(tokens) < shard_size:
        buffer[current_size:current_size+len(tokens)] = tokens
        current_size += len(tokens)
        if progress_bar is None:
          progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
      else:  # 缓冲区存放不下数据：把数据装入缓冲区最后部分，将缓冲区写入磁盘；然后把数据的剩余部分写入缓冲区起始。
        split = 'val' if shard_index == 0 else 'train'  #@ 第一个分片用于验证
        filename = Path(SHARD_DIR) / f"{shard_prefix}_{split}_{shard_index:06d}"
        remainder = shard_size - current_size   # 当前buffer的剩余空间
        progress_bar.update(remainder); progress_bar.close()
        buffer[current_size:current_size+remainder] = tokens[:remainder]
        write_datafile(filename, buffer)
        shard_index += 1
        # 填充新部分
        buffer[:len(tokens)-remainder] = tokens[remainder:]
        current_size = len(tokens) - remainder
        progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(current_size)

    if current_size != 0: # 处理最后的部分
      split = 'val' if shard_index == 0 else 'train'
      filename = Path(SHARD_DIR) / f"{shard_prefix}_{split}_{shard_index:06d}"
      write_datafile(filename, buffer[:current_size])
    
