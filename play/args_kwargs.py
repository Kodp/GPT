def f(*args, **kwargs):
  for i in args:
    print(i)
  for k, v in kwargs.items():
    print(k, v)


f(1,2,a=3,b=4,c=5,asdfljsadfsda="sssssssss")