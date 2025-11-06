# from build import soldier_gun

# s=soldier_gun.Soldier("liuzhihao")
# g=soldier_gun.Gun("AK47")
# tmp_count1=g._bullet_count

# s.addGun(g)
# s.addBulletToGun(20)
# tmp_count2=g._bullet_count
# s.fire()
# print(f"There is one soldier named {s._name}.")
# print(f"We give her one {g._type} which has {tmp_count1} bullet(s).")
# print(f"g._bullet_count={g._bullet_count}，tmp_count={tmp_count1}")
# print(f"She adds {tmp_count2-tmp_count1} bullet(s) to it.")
# print(f"After she fires the gun, it has {g._bullet_count} bullert(s) left.")

# void caul(){
#     std::cout<<"开始计算耗时函数"<<std::endl;
#     auto start = std::chrono::high_resolution_clock::now();
#     float sum = 0.0;
#     for (int i = 0; i < 2000000000; ++i) {
#         sum += std::sqrt(i); 
#     }
#     std::cout<< "Sum: " << sum << std::endl;
#     // 结束时间
#     auto end = std::chrono::high_resolution_clock::now();
    
#     // 计算持续时间
#     auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
#     std::cout << "耗时: " << duration.count() << " 微秒" << std::endl;
#     std::cout << "耗时: " << duration.count()/1e+6 << " 秒" << std::endl;
# }
# import time
# print('python--开始计算耗时函数')
# start=time.time()
# sum=0.0
# for i in range(2000000000):
#     sum += i**0.5
# end=time.time()
# print("Sum:",sum)
# print("耗时:",end-start,"秒")

# import numpy as np
# print('numpy--开始计算耗时函数')
# start=time.time()
# n = 2000000000
# # 创建数组并向量化计算
# arr = np.arange(n, dtype=np.float64)
# sum=np.sum(np.sqrt(arr))
# print("Sum:",sum)
# print("耗时:",end-start,"秒")
# from build import draw

# # draw.drw()
# draw.caul()
from build import draw as iter_test
import time
s=time.time()
iter_test.iter_main()
e=time.time()
print("pybind11耗时:",e-s,"秒")

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
print(f"python总耗时: {end_time - start_time} 秒")

# import sys
# import os

# build_path = '/home/tina/workspaces/soldiers_guns/build'
# sys.path.append(build_path)

# # 检查目录内容
# print("Build 目录内容:", os.listdir(build_path))

# # 查找 .so 文件
# for file in os.listdir(build_path):
#     if file.endswith('.so'):
#         print(f"找到 .so 文件: {file}")

# # 尝试导入
# try:
#     import draw
#     print("导入成功!")
#     draw.drw()
# except ImportError as e:
#     print(f"导入失败: {e}")
