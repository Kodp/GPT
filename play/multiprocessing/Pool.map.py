from multiprocessing import Pool

def square(x):
  return x * x

with Pool(processes=4) as pool:
  numbers = [1, 2, 3, 4, 5]
  results = pool.map(square, numbers)
  
  print("Map results:", results)