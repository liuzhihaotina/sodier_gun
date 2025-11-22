from collections import defaultdict
import time

# 创建测试数据
d = defaultdict(int)
for i in range(10000):
    d[f'key_{i}'] = i % 100

def test_methods():
    # 方法1: 生成器表达式
    start = time.time()
    total1 = sum(v for v in d.values())
    time1 = time.time() - start
    
    # 方法2: 直接sum
    start = time.time()
    total2 = sum(d.values())
    time2 = time.time() - start
    
    # 方法3: 使用循环
    start = time.time()
    total3 = 0
    for v in d.values():
        total3 += v
    time3 = time.time() - start
    
    print(f"数据量: {len(d)} 个键值对")
    print(f"总和: {total1}")
    print(f"方法1 (生成器): {time1:.6f}秒")
    print(f"方法2 (直接sum): {time2:.6f}秒") 
    print(f"方法3 (循环): {time3:.6f}秒")

test_methods()

# -------------------defualtdict--------------
# from collections import defaultdict

# # 错误理解：默认值在创建时生成
# dd = defaultdict(list)
# print(dd['nonexistent'])  # 这里才执行 list() 创建默认值

# # 验证：每次访问新键都会调用工厂函数
# counter = defaultdict(lambda: print("工厂函数被调用!"))
# print("第一次访问:")
# value1 = counter['key1']  # 会打印 "工厂函数被调用!"
# print("第二次访问不同键:")
# value2 = counter['key2']  # 会再次打印 "工厂函数被调用!"
# if 'key3' in counter:
#     print('key3 in counter')

# --------------------------SortedList update 批量添加元素,性能更高-----------
# from sortedcontainers import SortedList
# import time

# def test_performance():
#     data = list(range(10000))
    
#     # 方法1: 逐个添加
#     start = time.time()
#     sl1 = SortedList()
#     for num in data:
#         sl1.add(num)
#     time1 = time.time() - start
    
#     # 方法2: 使用 update 批量添加
#     start = time.time()
#     sl2 = SortedList()
#     sl2.update(data)
#     time2 = time.time() - start
    
#     print(f"逐个添加: {time1:.4f}秒")
#     print(f"批量update: {time2:.4f}秒")
#     print(f"性能提升: {time1/time2:.2f}倍")

# # test_performance()
# f = SortedList((2,3,1))
# # f.update((2,3,1))
# print(f)

# ----------------------------------SortedList bisect用法及性能-------------
# import time
# import bisect
# from sortedcontainers import SortedList

# def test_performance():
#     # 测试数据
#     test_data = list(range(100000, 0, -1))  # 倒序的10000个数字
    
#     # 方法1: 普通排序
#     start = time.time()
#     regular_list = []
#     for num in test_data:
#         regular_list.append(num)
#         regular_list.sort()
#     time1 = time.time() - start
    
#     # 方法2: bisect + 列表
#     start = time.time()
#     bisect_list = []
#     for num in test_data:
#         pos = bisect.bisect_left(bisect_list, num)
#         bisect_list.insert(pos, num)
#     time2 = time.time() - start
    
#     # 方法3: SortedList
#     start = time.time()
#     sorted_list = SortedList()
#     for num in test_data:
#         sorted_list.add(num)
#     time3 = time.time() - start
    
#     print(f"普通排序: {time1:.4f}秒")
#     print(f"bisect插入: {time2:.4f}秒") 
#     print(f"SortedList: {time3:.4f}秒")

# test_performance()

# -----------------------@cache--------------------------
# from functools import cache
# import time

# @cache
# def expensive_calculation(x, y):
#     print(f"计算 {x} + {y}...")  # 这行只会打印一次相同的参数组合
#     # 模拟耗时计算
#     result = 0
#     for i in range(100000000):
#         result += x * y
#     return result

# # 第一次调用会执行计算
# s = time.time()
# print(expensive_calculation(5, 10))
# print(f'第一次调用，运算时间: ', time.time()-s)
# # 第二次相同参数的调用会直接返回缓存结果
# s = time.time()
# print(expensive_calculation(5, 10))
# print(f'第二次调用，运算时间: ', time.time()-s)

# import numba
# import numpy as np
# import timeit

# def slow_sum(arr):
#     total = 0.0
#     for i in range(arr.shape[0]):
#         total += arr[i]
#     return total

# @numba.jit(nopython=True)
# def fast_sum(arr):
#     total = 0.0
#     for i in range(arr.shape[0]):
#         total += arr[i]
#     return total

# # 准备测试数据
# arr = np.random.rand(100000000)

# # 使用 timeit 进行性能测试
# print("测试 fast_sum (Numba加速):")
# fast_time = timeit.timeit(lambda: fast_sum(arr), number=10)
# print(f"10次运行平均时间: {fast_time/10:.4f} 秒")

# print("\n测试 slow_sum (原始Python):")
# slow_time = timeit.timeit(lambda: slow_sum(arr), number=10)
# print(f"10次运行平均时间: {slow_time/10:.4f} 秒")

# print(f"\n加速比: {slow_time/fast_time:.1f}x")