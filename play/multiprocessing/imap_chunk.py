import multiprocessing as mp
from time import sleep


def func(x):
  sleep(0.5)
  # print(f"Processing {x}")
  return x

X = [i for i in range(10)]

with mp.Pool(2) as pool:
  # 一次返回一个结果；一个进程处理chunksize个任务后再返回结果； 
  # 每一个进程的函数的参数都是X中的一个元素，换句话说就是单参数；在func里解包就可以做到多参数
  #fix still magic for me
  for res in pool.imap(func, X, chunksize=2):
    print(res)
