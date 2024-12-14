# 队列通信，n生产-n消费
import multiprocessing as mp

def producer(queue:mp.Queue):
  for i in range(5):
    queue.put(f"Data {i}")
    print(f"Produced: Data {i}\n",end="")

def consumer(queue:mp.Queue):
  while True:
    data = queue.get()
    if data is None:
      break
    print(f"Consumed: {data}\n", end="")


if __name__ == "__main__":
  queue = mp.Queue()   # 创建共享队列
  p1 = mp.Process(target=producer, args=(queue,))
  p2 = mp.Process(target=consumer, args=(queue,))
  
  p1.start()
  p2.start()
  
  p1.join()
  queue.put(None)
  p2.join()