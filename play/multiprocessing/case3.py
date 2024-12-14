import multiprocessing
import os

print(os.getpid())

def f():
  print(os.getpid())
  
p = multiprocessing.Process(target=f)
p.start()
p.join()


# 第二种用法：继承并重载run函数
class MyProcess(multiprocessing.Process):
  def run(self):
    f()
  
p = MyProcess()
p.start()
p.join()