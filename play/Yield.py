def my_generator():
    yield 1
    yield 2
    yield 3

gen = my_generator()  # 调用生成器函数，返回生成器对象
print(gen)  # 输出：<generator object my_generator at 0x...>

# 生成器对象是迭代器
print(next(gen))  # 输出：1
print(next(gen))  # 输出：2
print(next(gen))  # 输出：3
# print(next(gen))  # 再调用会抛出 StopIteration 异常
