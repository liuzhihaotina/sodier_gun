import time
numbers = list(range(1, 100000001))
it = iter(numbers)
start_time = time.time()
while True:
    try:
        a = next(it)
        if a % 10000000 == 0:
            print(f"已迭代到: {a} ", end='')  # , flush=True
    except StopIteration:
        print("\n迭代结束")
        break
end_time = time.time()
print(f"总耗时: {end_time - start_time} 秒")
