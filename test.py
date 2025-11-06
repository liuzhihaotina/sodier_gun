import numpy as np
import time
print('numpy--开始计算耗时函数')
start=time.time()
total_sum = 0.0
n=2000000000; batch_size=10000000
batches = n // batch_size

for batch_num in range(batches + 1):
    s = batch_num * batch_size
    e = min((batch_num + 1) * batch_size, n)
    
    if s < e:
        batch = np.arange(s, e, dtype=np.float64)
        total_sum += np.sum(np.sqrt(batch))
end=time.time()
print("Sum:",total_sum)
print("耗时:",end-start,"秒")
