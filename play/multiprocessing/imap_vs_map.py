from multiprocessing import Pool
import time

# 模拟一个耗时任务
def slow_task(x):
  # print(f"{x:02} sleep 0.5s\n",end='')
  time.sleep(0.2)  # 模拟任务耗时
  return x

if __name__ == "__main__":
  numbers = list(range(50))  # 大量任务

  #   # 使用 map
  # start = time.time()
  # with Pool(processes=4) as pool:
  #   results = pool.map(slow_task, numbers)  # 阻塞直到所有任务完成
  # print(results)
  # print("Map Time:", time.time() - start)

  # 使用 imap
  start = time.time()
  with Pool(processes=4) as pool:
    result_iter = pool.imap(slow_task, numbers, chunksize=3)
    for i, result in enumerate(result_iter):  # 可以边取边处理
      print(i, result)
  print("iMap Time:", time.time() - start)
