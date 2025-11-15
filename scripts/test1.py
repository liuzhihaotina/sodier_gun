import numba
import numpy as np
import timeit

def slow_sum(arr):
    total = 0.0
    for i in range(arr.shape[0]):
        total += arr[i]
    return total

@numba.jit(nopython=True)
def fast_sum(arr):
    total = 0.0
    for i in range(arr.shape[0]):
        total += arr[i]
    return total

# 准备测试数据
arr = np.random.rand(100000000)

# 使用 timeit 进行性能测试
print("测试 fast_sum (Numba加速):")
fast_time = timeit.timeit(lambda: fast_sum(arr), number=10)
print(f"10次运行平均时间: {fast_time/10:.4f} 秒")

print("\n测试 slow_sum (原始Python):")
slow_time = timeit.timeit(lambda: slow_sum(arr), number=10)
print(f"10次运行平均时间: {slow_time/10:.4f} 秒")

print(f"\n加速比: {slow_time/fast_time:.1f}x")