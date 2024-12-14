# 基本多进程, 每个进程计算一个函数，无通信
import multiprocessing as mp
from multiprocessing import process

def worker(num):
  print(f"Worker {num} running")

if __name__ == "__main__":
  print(__file__)
  processes = []
  for i in range(4):  # 创建4个子进程
    p = mp.Process(target=worker, args=(i,)) 
    processes.append(p)
    p.start()
  
  for p in processes:
    p.join()# 等待所有子进程完成
  