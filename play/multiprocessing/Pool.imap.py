from multiprocessing import Pool

def square(x):
  return x * x

with Pool(processes=4) as pool:
  numbers = [1, 2, 3, 4, 5]
  result_iter = pool.imap(square, numbers) # 返回的不是结果，而是结果的迭代器
  
  print("imap results:")
  for result in result_iter:
    print(result)
  