# -----------------------------------------------------
# import torch
# d = torch.tensor([[1,0.1],[2,0.2],[3,0.3],[1,0.4],[2,0.5],[3,0.6]])
# d_c = d.view(-1,3,d.shape[1])
# pass
# -------------------time_series bike_data-------------------------------
# import torch
# import numpy as np

# bikes_numpy = np.loadtxt(
# "/mnt/d/书籍/训练模型/dlwpt-code-master/data/p1ch4/bike-sharing-dataset/hour-fixed.csv",
# dtype=np.float32,
# delimiter=",",
# skiprows=1,
# converters={1: lambda x: float(x[8:10])})
# bikes = torch.from_numpy(bikes_numpy)
# first_day = bikes[:24].long()
# weather_onehot = torch.zeros(first_day.shape[0], 4)
# weather_onehot.scatter_(
# dim=1,
# index=first_day[:,9].unsqueeze(1).long() - 1, value=1.0)
# pass 

# ------------------------------------np.float16/32/64内存占用对比-------------------------------------
# import numpy as np
# def memory_comparison():
#     print("内存占用对比:")
#     print("=" * 40)

#     size = 1000000  # 100万个元素

#     arr_16 = np.ones(size, dtype=np.float16)
#     arr_32 = np.ones(size, dtype=np.float32)
#     arr_64 = np.ones(size, dtype=np.float64)

#     print(f"float16 数组: {arr_16.nbytes / 1024:.2f} KB")
#     print(f"float32 数组: {arr_32.nbytes / 1024:.2f} KB")
#     print(f"float64 数组: {arr_64.nbytes / 1024:.2f} KB")
#     print()
#     print(f"float16 只有 float32 的 {arr_16.nbytes/arr_32.nbytes:.0%}")
#     print(f"float16 只有 float64 的 {arr_16.nbytes/arr_64.nbytes:.0%}")

# memory_comparison()

# ------------------------------读取csv文件为numpy数组-----------------------------
# import numpy as np
# import csv
# import torch
# # 设置打印选项，禁用科学计数法
# torch.set_printoptions(sci_mode=False)

# wine_path = "/mnt/d/书籍/训练模型/dlwpt-code-master/data/p1ch4/tabular-wine/winequality-white.csv"
# wineq_numpy = np.loadtxt(wine_path, dtype=np.float32,
#                          delimiter=";", skiprows=1)
# a = csv.reader(open(wine_path), delimiter=';')
# # print(f"a.type={type(a)}")
# col_list = next(a)
# # print(wineq_numpy.shape) # (4898, 12)
# # print(col_list)
# # print(wineq_numpy[0])  # 查看第一行数据
# # print(next(a))
# # wineq_numpy2 = np.array(list(a), dtype=np.float32)
# wineq_tensor = torch.from_numpy(wineq_numpy)
# # print(wineq_tensor.shape, wineq_tensor.dtype, wineq_tensor.device)
# data = wineq_tensor[:, :-1]
# data_mean = data.mean(dim=0)
# data_var = data.var(dim=0)
# # print(data_mean.shape, data_var.shape)
# data_normalized = (data - data_mean) / data_var.sqrt()
# # print('---data：', data[0])#.to(dtype=torch.int))
# # print('---data that be normalized：', data_normalized[0])

# # print('----float16:',data[0].half().dtype)
# # print('----float32:',data[0].float().dtype)
# # print('----float64:',data[0].double().dtype)
# target = wineq_tensor[:, -1].long()
# # print(target.shape, target.shape[0], type(target.shape), type(target.shape[0]))
# # tager_onehot = torch.nn.functional.one_hot(target, num_classes=10)
# target_onehot = torch.zeros(target.shape[0], 10).scatter_( 1, target.unsqueeze(1), 1.0)

# # bad_index = (target <= 3).nonzero(as_tuple=True)[0] # 找出满足条件的索引
# bad_mask = (target <= 3) # 布尔掩码，满足条件的True，否则False
# mid_mask = (target > 3) & (target < 7)
# good_mask = (target >= 7)
# bad_data = data[bad_mask]
# mid_data = data[mid_mask]
# good_data = data[good_mask]

# bad_mean = bad_data.mean(dim=0)
# mid_mean = mid_data.mean(dim=0)
# good_mean = good_data.mean(dim=0)

# for i, args in enumerate(zip(col_list[:-1], bad_mean, mid_mean, good_mean)):
#     print('{:2} {:20} {:6.2f} {:6.2f} {:6.2f}'.format(i, *args))

# pass
# bad_data = data_normalized[bad_mask]
# mid_data = data_normalized[mid_mask]
# good_data = data_normalized[good_mask]
# print(bad_index.shape, bad_mask.shape)
# print('------', bad_mask.sum().item())
# print(bad_index)
# print(target.shape)
# print(target.min(),target.max())
# print(torch.equal(tager_onehot, target_onehot))
# print((torch.eq(tager_onehot, target_onehot)).all())

# import torch

# data = torch.tensor([
#     [1, 2, 3],
#     [4, 5, 6], 
#     [7, 8, 9]
# ])

# # 多条件筛选：值大于3且小于8
# condition = (data > 3) & (data < 8)
# indices = condition.nonzero(as_tuple=True)
# i=condition.nonzero()

# print("原始数据:")
# print(data)
# print(f"\n满足条件的位置: {indices}")
# print(f"满足条件的值: {data[indices]}")

# --------------------迭代器的使用-next------------------------
# # 创建列表迭代器
# # my_list = [1, 2, 3, 4, 5]
# my_list = (1,2,3,4,5)
# iterator = iter(my_list)

# print(next(iterator))  # 输出: 1
# print(next(iterator))  # 输出: 2
# print(next(iterator))  # 输出: 3
# print(next(iterator))  # 输出: 4
# print(next(iterator))  # 输出: 5
# print(tuple(iterator))
# # print(next(iterator))  # 如果取消注释，会抛出 StopIteration 异常

# --------------------计算不同h/w图片的均值、标准差-------------------------------
# import os
# import torch
# import imageio.v2 as imageio
# batch_size = 4

# data_dir = 'data/test/cats'
# filenames = [name for name in os.listdir(
#     data_dir) if os.path.splitext(name)[-1] == '.jpg']
# # print(len(os.listdir(data_dir)), len(filenames))
# shapes = set()
# for i, filename in enumerate(filenames):
#     img_arr = imageio.imread(os.path.join(data_dir, filename))
#     img_t = torch.from_numpy(img_arr)
#     img_t = img_t.permute(2, 0, 1)
#     img_t = img_t[:3]
#     shapes.add(tuple(img_t.shape[1:])+(filename,))
# # print(shapes)
# print(sorted(list(shapes), key=lambda x: (x[0], x[1])))
# import torch
# import torchvision.transforms as transforms
# from PIL import Image

# def calculate_mean_std_torch(image_paths):
#     """
#     使用PyTorch计算均值和标准差
#     """
#     channel_sum = torch.zeros(3)
#     channel_sum_squared = torch.zeros(3)
#     pixel_num = 0

#     transform = transforms.ToTensor()  # 自动归一化到 [0, 1]

#     for img_path in image_paths:
#         img = Image.open("data/test/cats/"+img_path).convert('RGB')
#         img_tensor = transform(img)  # 形状: (3, H, W)

#         # 累加统计量
#         pixel_num += img_tensor.shape[1] * img_tensor.shape[2]
#         channel_sum += torch.sum(img_tensor, dim=(1, 2))
#         channel_sum_squared += torch.sum(img_tensor**2, dim=(1, 2))

#     # 计算均值和标准差
#     mean = channel_sum / pixel_num
#     std = torch.sqrt(channel_sum_squared / pixel_num - mean**2)

#     return mean.numpy(), std.numpy()

# # 使用示例
# mean, std = calculate_mean_std_torch(os.listdir("data/test/cats"))
# print(f"均值 (R, G, B): {mean}")
# print(f"标准差 (R, G, B): {std}")


# --------------------------PG  vs  PNG 文件大小对比测试---------------
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import os
# import numpy as np

# def compare_jpg_png_sizes():
#     print("=" * 60)
#     print("JPG  vs  PNG 文件大小对比测试")
#     print("=" * 60)

#     # 1. 创建测试图像数据
#     print("\n1. 创建测试图像数据")

#     # 场景1: 随机噪声图像（高熵，难以压缩）
#     random_tensor = torch.rand(3, 224, 224)
#     random_pil = transforms.ToPILImage()(random_tensor)

#     # 场景2: 平滑渐变图像（低熵，容易压缩）
#     x = torch.linspace(0, 1, 224).view(1, 224, 1)
#     y = torch.linspace(0, 1, 224).view(1, 1, 224)
#     gradient_tensor = (x + y) / 2
#     gradient_tensor = gradient_tensor.repeat(3, 1, 1)
#     gradient_pil = transforms.ToPILImage()(gradient_tensor)

#     # 场景3: 真实图像模拟（包含边缘和纹理）
#     real_like_tensor = torch.rand(3, 224, 224)
#     # 添加一些边缘效果
#     real_like_tensor[:, 100:120, :] = 0.8  # 水平条带
#     real_like_tensor[:, :, 100:120] = 0.3  # 垂直条带
#     real_like_pil = transforms.ToPILImage()(real_like_tensor)

#     test_cases = [
#         ("随机噪声", random_pil),
#         ("平滑渐变", gradient_pil),
#         ("模拟真实", real_like_pil)
#     ]

#     results = []

#     # 2. 测试不同质量设置的JPG
#     print("\n2. 测试不同质量设置的JPG")
#     jpg_qualities = [30, 50, 75, 95]

#     for name, pil_image in test_cases:
#         print(f"\n--- {name}图像 ---")
#         case_results = {'name': name}

#         # 保存PNG
#         png_path = f'{name.lower().replace(" ", "_")}.png'
#         pil_image.save(png_path, format='PNG', optimize=True)
#         png_size = os.path.getsize(png_path)
#         case_results['png'] = png_size

#         # 保存不同质量的JPG
#         jpg_sizes = {}
#         for quality in jpg_qualities:
#             jpg_path = f'{name.lower().replace(" ", "_")}_q{quality}.jpg'
#             pil_image.save(jpg_path, format='JPEG', quality=quality, optimize=True)
#             jpg_sizes[quality] = os.path.getsize(jpg_path)

#         case_results['jpg'] = jpg_sizes
#         results.append(case_results)

#         # 打印结果
#         print(f"PNG 文件大小: {png_size/1024:.1f} KB")
#         for quality, size in jpg_sizes.items():
#             ratio = size / png_size
#             print(f"JPG Q{quality:2d}: {size/1024:6.1f} KB ({ratio:.1%} of PNG)")

#     return results, test_cases

# def test_grayscale_images():
#     print("\n" + "=" * 60)
#     print("灰度图像文件大小对比")
#     print("=" * 60)

#     # 创建灰度测试图像
#     gray_tensor = torch.rand(1, 224, 224)  # 单通道灰度
#     gray_pil = transforms.ToPILImage()(gray_tensor)

#     # 保存为PNG
#     gray_pil.save('grayscale_test.png', format='PNG', optimize=True)
#     png_size = os.path.getsize('grayscale_test.png')

#     # 保存为不同质量的JPG
#     gray_sizes = {}
#     for quality in [30, 50, 75, 95]:
#         gray_pil.save(f'grayscale_test_q{quality}.jpg', format='JPEG', quality=quality, optimize=True)
#         gray_sizes[quality] = os.path.getsize(f'grayscale_test_q{quality}.jpg')

#     print(f"灰度PNG: {png_size/1024:.1f} KB")
#     for quality, size in gray_sizes.items():
#         ratio = size / png_size
#         print(f"灰度JPG Q{quality}: {size/1024:.1f} KB ({ratio:.1%} of PNG)")

#     return png_size, gray_sizes

# def test_compression_ratios():
#     print("\n" + "=" * 60)
#     print("压缩率分析")
#     print("=" * 60)

#     # 计算理论最大压缩率
#     uncompressed_size = 3 * 224 * 224  # 3通道, 224x224, 每像素1字节
#     print(f"未压缩数据大小: {uncompressed_size/1024:.1f} KB")

#     # 测试实际图像
#     test_image = torch.rand(3, 224, 224)
#     test_pil = transforms.ToPILImage()(test_image)

#     # 无优化保存
#     test_pil.save('test_no_optimize.png', format='PNG', optimize=False)
#     test_pil.save('test_no_optimize.jpg', format='JPEG', quality=95, optimize=False)

#     # 优化保存
#     test_pil.save('test_optimize.png', format='PNG', optimize=True)
#     test_pil.save('test_optimize.jpg', format='JPEG', quality=95, optimize=True)

#     sizes = {
#         'PNG无优化': os.path.getsize('test_no_optimize.png'),
#         'PNG优化': os.path.getsize('test_optimize.png'),
#         'JPG无优化': os.path.getsize('test_no_optimize.jpg'),
#         'JPG优化': os.path.getsize('test_optimize.jpg')
#     }

#     print("优化效果对比:")
#     for name, size in sizes.items():
#         ratio = size / uncompressed_size
#         print(f"{name}: {size/1024:.1f} KB ({ratio:.1%} of 未压缩)")

# # 运行测试
# if __name__ == "__main__":
#     # 主要测试
#     results, test_cases = compare_jpg_png_sizes()

#     # 灰度图像测试
#     gray_png, gray_jpg = test_grayscale_images()

#     # 压缩率测试
#     test_compression_ratios()

#     # 清理测试文件
#     print("\n清理测试文件...")
#     for name, _ in test_cases:
#         base_name = name.lower().replace(" ", "_")
#         # 删除PNG
#         if os.path.exists(f'{base_name}.png'):
#             os.remove(f'{base_name}.png')
#         # 删除JPG
#         for quality in [30, 50, 75, 95]:
#             jpg_file = f'{base_name}_q{quality}.jpg'
#             if os.path.exists(jpg_file):
#                 os.remove(jpg_file)

#     # 删除其他测试文件
#     for file in ['grayscale_test.png', 'grayscale_test_q30.jpg', 'grayscale_test_q50.jpg',
#                  'grayscale_test_q75.jpg', 'grayscale_test_q95.jpg', 'test_no_optimize.png',
#                  'test_no_optimize.jpg', 'test_optimize.png', 'test_optimize.jpg']:
#         if os.path.exists(file):
#             os.remove(file)

#  -----------------------------tensor数学操作------------------------
# import torch
# import math

# print("=" * 50)
# print("数学操作练习: 平方根 (square root)")
# print("=" * 50)

# # 创建测试张量
# a = torch.tensor(list(range(9)))
# print(f"原始张量 a = {a}")

# print("\n" + "=" * 50)
# print("步骤 a: 应用平方根函数到 a")
# print("=" * 50)

# # a. 应用平方根函数 element-wise 到 a
# try:
#     result = torch.sqrt(a)
#     print(f"torch.sqrt(a) = {result}")
# except Exception as e:
#     print(f"错误类型: {type(e).__name__}")
#     print(f"错误信息: {e}")

# print("\n" + "=" * 50)
# print("步骤 b: 使函数工作所需的操作")
# print("=" * 50)

# # b. 使函数工作所需的操作
# print("解决方案 1: 转换为浮点数")
# a_float = a.float()
# result_float = torch.sqrt(a_float)
# print(f"a.float() = {a_float}")
# print(f"torch.sqrt(a.float()) = {result_float}")

# print("\n解决方案 2: 使用 torch.sqrt() 的 out 参数")
# result_out = torch.empty_like(a, dtype=torch.float32)
# torch.sqrt(a.float(), out=result_out)
# print(f"使用 out 参数的结果 = {result_out}")

# print("\n" + "=" * 50)
# print("步骤 c: 查找原地操作版本")
# print("=" * 50)

# # c. 查找原地操作版本
# print("检查原地操作版本:")

# # 方法 1: 直接尝试 sqrt_
# try:
#     a_copy = a.clone().float()  # 创建副本并转换为float
#     print(f"操作前: a_copy = {a_copy}")
#     a_copy.sqrt_()
#     print(f"a_copy.sqrt_() 后: {a_copy}")
#     print("✅ torch.Tensor.sqrt_() 存在并工作")
# except Exception as e:
#     print(f"❌ sqrt_ 不可用: {e}")

# # 方法 2: 检查其他数学函数的原地版本
# print("\n其他数学函数的原地操作示例:")
# b = torch.tensor([1.0, 4.0, 9.0, 16.0])
# print(f"原始: b = {b}")

# # 余弦函数的原地操作
# b_cos = b.clone()
# b_cos.cos_()
# print(f"b.cos_() = {b_cos}")

# # 正弦函数的原地操作
# b_sin = b.clone()
# b_sin.sin_()
# print(f"b.sin_() = {b_sin}")

# # 指数函数的原地操作
# b_exp = b.clone()
# b_exp.exp_()
# print(f"b.exp_() = {b_exp}")

# -----------------------------------------------------storage、view---------------------------------------------------------
# import torch

# print("=" * 50)
# print("步骤 1: 创建张量 a")
# print("=" * 50)

# # 创建张量 a
# a = torch.tensor(list(range(9)))
# print(f"a = {a}")
# print(f"a 的存储: {a.storage()}")

# # 检查 a 的属性
# print(f"a.size() = {a.size()}")
# print(f"a.storage_offset() = {a.storage_offset()}")
# print(f"a.stride() = {a.stride()}")

# print("\n" + "=" * 50)
# print("步骤 2: 使用 view 创建张量 b")
# print("=" * 50)

# # 创建 b = a.view(3, 3)
# b = a.view(3, 3)
# print(f"b = \n{b}")

# # 检查 b 的属性
# print(f"b.size() = {b.size()}")
# print(f"b.storage_offset() = {b.storage_offset()}")
# print(f"b.stride() = {b.stride()}")

# # 验证 a 和 b 是否共享存储
# print(f"\na 和 b 是否共享存储: {a.storage().data_ptr() == b.storage().data_ptr()}")
# b[0, 0] = 999
# print(f"修改 b 会影响 a: b[0, 0] = 999; a = {a}")  # 验证共享存储

# print("\n" + "=" * 50)
# print("步骤 3: 创建张量 c = b[1:, 1:]")
# print("=" * 50)

# # 创建 c = b[1:, 1:]
# c = b[1:, 1:]
# print(f"c = \n{c}")

# # 检查 c 的属性
# print(f"c.size() = {c.size()}")
# print(f"c.storage_offset() = {c.storage_offset()}")
# print(f"c.stride() = {c.stride()}")

# print("\n" + "=" * 50)
# print("验证所有张量是否共享存储")
# print("=" * 50)

# # 验证所有张量是否共享同一存储
# print(f"a.storage() 地址: {a.storage().data_ptr()}")
# print(f"b.storage() 地址: {b.storage().data_ptr()}")
# print(f"c.storage() 地址: {c.storage().data_ptr()}")
# print(f"所有张量共享存储: {a.storage().data_ptr() == b.storage().data_ptr() == c.storage().data_ptr()}")

# ----------------------------------.pt跟.hdf5保存的性能对比，.pt写和加载都更快--------------------------------------
# import time
# import torch
# import h5py
# import numpy as np

# def performance_comparison():
#     """性能对比测试"""
#     large_tensor = torch.randn(5000, 5000)  # 约 100MB 数据

#     # .pt 格式测试
#     start_time = time.time()
#     torch.save(large_tensor, 'tmp/test.pt')
#     pt_save_time = time.time() - start_time

#     start_time = time.time()
#     loaded_pt = torch.load('tmp/test.pt')
#     pt_load_time = time.time() - start_time

#     # .hdf5 格式测试
#     start_time = time.time()
#     with h5py.File('tmp/test.hdf5', 'w') as f:
#         f.create_dataset('data', data=large_tensor.numpy(), compression='gzip')
#     h5_save_time = time.time() - start_time

#     start_time = time.time()
#     with h5py.File('tmp/test.hdf5', 'r') as f:
#         loaded_array = f['data'][:]
#     loaded_h5 = torch.from_numpy(loaded_array)
#     h5_load_time = time.time() - start_time

#     print(f".pt 格式 - 保存: {pt_save_time:.3f}s, 加载: {pt_load_time:.3f}s")
#     print(f".hdf5格式 - 保存: {h5_save_time:.3f}s, 加载: {h5_load_time:.3f}s")

# performance_comparison()


# ------------------------------不同精度类型的tensor占用内存对比----------------------------------
# import torch
# def check_memory_usage():
#     """检查不同数据类型的记忆体占用"""
#     size = 1000  # 1000个元素

#     tensor_f32 = torch.randn(size, size, dtype=torch.float32)
#     tensor_f64 = torch.randn(size, size, dtype=torch.float64)

#     print(f"float32 张量内存: {tensor_f32.element_size() * tensor_f32.nelement() / 1024**2:.2f} MB")
#     print(f"float64 张量内存: {tensor_f64.element_size() * tensor_f64.nelement() / 1024**2:.2f} MB")
#     print(f"内存比例: {tensor_f64.element_size() / tensor_f32.element_size():.1f}x")

# check_memory_usage()


# -----------------------------灰度处理图片、命名tensor使用----------------------------------------
# import torch
# from PIL import Image
# from torchvision import models
# from torchvision import transforms
# import matplotlib.pyplot as plt

# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )])
# # weights = torch.tensor([0.2126, 0.7152, 0.0722])
# # img = Image.open(f"data/test/dogs/dog.4001.jpg")
# # img_t = preprocess(img)
# # tensor_imgs = img_t
# # for i in range(2,3):
# #     img = Image.open(f"data/test/dogs/dog.400{i}.jpg")
# #     # img.show()
# #     # print(img_t.shape)
# #     tensor_imgs = torch.stack([tensor_imgs, preprocess(img)], dim=0)

# # # unsqueezed_weights = weights.unsqueeze(-1).unsqueeze_(-1)
# # # img_weights = (img_t * unsqueezed_weights)
# # # batch_weights = (tensor_imgs * unsqueezed_weights)
# # # img_gray_weighted = img_weights.sum(-3)
# # # batch_gray_weighted = batch_weights.sum(-3)
# # # print(img_gray_weighted.shape, batch_gray_weighted.shape, unsqueezed_weights.shape)
# # img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
# # batch_named = tensor_imgs.refine_names(..., 'batch', 'channels', 'rows', 'columns')
# # # print("img named:", img_named.shape, img_named.names)
# # # print("batch named:", batch_named.shape, batch_named.names)
# # weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
# # weights_aligned = weights_named.align_as(img_named)
# # # print(weights_aligned.shape, weights_aligned.names)
# # gray_named = (img_named * weights_aligned).sum('channels')
# # # print(gray_named.shape, gray_named.names)
# # gray_plain = gray_named.rename(None)
# # # print(gray_plain.shape)


# def visualize_normalized_image(original_tensor, gray_tensor, save_path="data/compare_normalized.png"):
#     """处理标准化后的图像数据"""

#     # 方法1: 反标准化（如果您知道原始标准化参数）
#     # 假设使用的是ImageNet标准化的均值和标准差
#     # [0.2126, 0.7152, 0.0722]
#     IMAGENET_MEAN = [0.485, 0.456, 0.406]
#     IMAGENET_STD = [0.229, 0.224, 0.225]

#     # 反标准化原始图像
#     original_denormalized = original_tensor.clone()
#     for i in range(3):
#         original_denormalized[i] = original_denormalized[i] * IMAGENET_STD[i] + IMAGENET_MEAN[i]

#     # 限制值范围在[0,1]之间
#     original_denormalized = torch.clamp(original_denormalized, 0, 1)

#     # 可视化
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     # 原始图像（反标准化后）
#     axes[0].imshow(original_denormalized.permute(1, 2, 0))
#     axes[0].set_title('Original Image (Denormalized)')
#     axes[0].axis('off')

#     # 灰度图像 - 需要归一化到[0,1]范围
#     gray_normalized = (gray_tensor - gray_tensor.min()) / (gray_tensor.max() - gray_tensor.min())
#     axes[1].imshow(gray_normalized, cmap='gray')
#     axes[1].set_title('Weighted Grayscale')
#     axes[1].axis('off')

#     plt.tight_layout()
#     # plt.show()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#     print(f"对比图已保存至: {save_path}")

#     # 关闭plt，避免内存泄漏
#     plt.close()

#     return gray_tensor

# def save_grayscale_image(gray_tensor, save_path):
#     """保存灰度图片，处理标准化数据"""
#     # 将灰度图归一化到[0,1]范围
#     gray_normalized = (gray_tensor - gray_tensor.min()) / (gray_tensor.max() - gray_tensor.min())

#     # 转换为PIL图像
#     gray_pil = transforms.ToPILImage()(gray_normalized.unsqueeze(0))

#     # 保存图片
#     gray_pil.save(save_path)
#     print(f"灰度图片已保存至: {save_path}")

# # 5. 主函数
# def main(image_path, save_path):
#     """主流程函数"""
#     # 加载图片
#     img = Image.open(image_path)
#     img_t = preprocess(img)
#     print(f"原始图片形状: {img_t.shape}")

#     # 灰度化处理
#     weights_named = torch.tensor([0.2126, 0.7152, 0.0722], names=['channels'])
#     img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
#     weights_aligned = weights_named.align_as(img_named)
#     gray_named = (img_named * weights_aligned).sum('channels')
#     gray_t = gray_named.rename(None)
#     print(f"灰度图形状: {gray_t.shape}")
#     print(f"灰度图值范围: [{gray_t.min():.3f}, {gray_t.max():.3f}]")

#     # 可视化对比
#     visualize_normalized_image(img_t, gray_t)

#     # 保存灰度图片
#     save_grayscale_image(gray_t, save_path)

#     return gray_t

# # 使用示例
# if __name__ == "__main__":
#     # 替换为您的图片路径
#     image_path = f"data/test/dogs/dog.4001.jpg"
#     save_path = "data/gray_dog4001.jpg"
#     gray_tensor = main(image_path, save_path)

# ------------------------------------图片预处理、预训练模型的推理------------------------------------------
# import torch
# from PIL import Image
# from torchvision import models
# from torchvision import transforms

# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )])
# img = Image.open("data/test/dogs/dog.4001.jpg")
# # img.show()
# img_t = preprocess(img)

# # 反标准化
# def denormalize(tensor, mean, std):
#     """反标准化张量"""
#     mean = torch.tensor(mean).view(-1, 1, 1)
#     std = torch.tensor(std).view(-1, 1, 1)
#     return tensor * std + mean


# # 反标准化并转换回PIL图像
# denormalized_img = denormalize(
#     img_t, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# denormalized_img = torch.clamp(denormalized_img, 0, 1)  # 限制在0-1范围内
# denormalized_img = transforms.ToPILImage()(denormalized_img)

# # 保存图片
# denormalized_img.save("data/preprocessed_dog40004.jpg")

# batch_t = torch.unsqueeze(img_t, 0)

# # print(dir(models))
# # resnet = models.resnet101(pretrained=True)
# with open('data/imagenet_classes.txt') as f:
#     labels = [line.strip() for line in f.readlines()]
# resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
# # print(resnet)
# resnet.eval()
# out = resnet(batch_t)
# # print(out.shape)
# _, index = torch.max(out, 1)
# percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# a, b = labels[index[0]], percentage[index[0]].item()
# print(f'Predicted: {a}, {b}%')

# _, indices = torch.sort(out, descending=True)
# c = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
# for i,j in c:
#     print(i, j)
# ------------------------------------------------------------------------------

# #!/usr/bin/env python3
# """
# LMDB reader + DALI pipeline (改进版)

# - LMDBReader: 自动识别单文件 .lmdb 或目录格式
# - DALILMDBPipeline:
#   - 在 define_graph 中创建 ExternalSource 的 DataNode 并保存为 self._jpegs / self._labels
#   - 在 iter_setup 中对这些 DataNode 调用 feed_input(...)
#   - 支持 decode_device 参数（'mixed'|'gpu'|'cpu'），若无 GPU 自动回退到 cpu
# """
# # import os
# # import lmdb
# # import json
# # import struct
# # import random
# # import numpy as np
# # from threading import Lock
# # import warnings

# # # DALI imports
# # from nvidia import dali
# # from nvidia.dali import pipeline as dali_pipeline
# # import nvidia.dali.fn as fn
# # import nvidia.dali.types as types
# # import nvidia.dali.ops as ops

# # Optional pynvml for more robust GPU detection
# try:
#     import pynvml
#     _HAS_PYNVML = True
# except Exception:
#     _HAS_PYNVML = False


# # class LMDBReader:
# #     """轻量的 LMDB reader，逐条返回 (image_bytes, label)
# #     兼容两种 LMDB 布局：
# #       - 单文件形式（.lmdb 文件） -> subdir=False
# #       - 目录形式（含 data.mdb/lock.mdb 等） -> subdir=True
# #     """
# #     def __init__(self, lmdb_path, shuffle=True):
# #         subdir = os.path.isdir(lmdb_path)
# #         env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, max_readers=32, subdir=subdir)
# #         self.env = env
# #         with self.env.begin() as txn:
# #             keys_blob = txn.get(b'__keys__')
# #             if keys_blob:
# #                 self.keys = json.loads(keys_blob.decode('utf-8'))
# #             else:
# #                 cur = txn.cursor()
# #                 self.keys = [k.decode('ascii') for k, _ in cur if not k.startswith(b'__')]
# #         self.shuffle = shuffle
# #         if self.shuffle:
# #             random.shuffle(self.keys)
# #         self._idx = 0
# #         self._lock = Lock()

# #     def __len__(self):
# #         return len(self.keys)

# #     def _read_item(self, key):
# #         with self.env.begin() as txn:
# #             raw = txn.get(key.encode('ascii'))
# #         if raw is None:
# #             raise KeyError("Key not found: {}".format(key))
# #         label = struct.unpack('>I', raw[:4])[0]
# #         img = raw[4:]
# #         return img, label

# #     def next(self):
# #         with self._lock:
# #             key = self.keys[self._idx]
# #             img, label = self._read_item(key)
# #             self._idx += 1
# #             if self._idx >= len(self.keys):
# #                 self._idx = 0
# #                 if self.shuffle:
# #                     random.shuffle(self.keys)
# #         return img, label

# #     def next_batch(self, batch_size):
# #         imgs = []
# #         labels = []
# #         for _ in range(batch_size):
# #             img, label = self.next()
# #             imgs.append(np.frombuffer(img, dtype=np.uint8))
# #             labels.append(np.array(label, dtype=np.int64))
# #         return imgs, np.stack(labels)


# def _detect_gpu_available():
#     """尝试检测是否有可用 NVIDIA GPU/可调用 NVML。"""
#     try:
#         if _HAS_PYNVML:
#             pynvml.nvmlInit()
#             count = pynvml.nvmlDeviceGetCount()
#             pynvml.nvmlShutdown()
#             return count > 0
#     except Exception:
#         pass
#     try:
#         import subprocess
#         p = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=2)
#         if p.returncode == 0 and p.stdout.strip():
#             return True
#     except Exception:
#         pass
#     return False
# print(_detect_gpu_available())

# class DALILMDBPipeline(dali_pipeline.Pipeline):
#     """
#     DALI pipeline that takes two ExternalSource inputs:
#       - jpegs (list of numpy uint8 arrays)
#       - labels (numpy int64 array)
#     Returns: images (float32, normalized), labels (int64)

#     decode_device: 'mixed' | 'gpu' | 'cpu'
#     """
#     def __init__(self, batch_size, num_threads, device_id, lmdb_reader,
#                  image_size=224, decode_device='mixed',
#                  mean=(0.485*255, 0.456*255, 0.406*255),
#                  std=(0.229*255, 0.224*255, 0.225*255)):
#         super().__init__(batch_size, num_threads, device_id, seed=12)
#         # ExternalSource operators
#         self.input_jpegs = ops.ExternalSource()
#         self.input_labels = ops.ExternalSource()

#         # 保存 DataNode 引用（将在 define_graph 中赋值）
#         self._jpegs = None
#         self._labels = None

#         # decode device selection with detection
#         self.requested_decode_device = decode_device
#         actual_device = decode_device
#         if decode_device in ('mixed', 'gpu'):
#             if not _detect_gpu_available():
#                 warnings.warn(
#                     f"Requested decode_device='{decode_device}' but no usable NVIDIA GPU / NVML found. "
#                     "Falling back to 'cpu' decode."
#                 )
#                 actual_device = 'cpu'
#         self.decode_device = actual_device

#         # ImageDecoder & postprocess
#         self.decode = ops.ImageDecoder(device=self.decode_device, output_type=types.RGB)
#         # Resize & normalization: put on GPU for performance if GPU present (device_id >= 0)
#         self.resize = ops.Resize(device='gpu', resize_shorter=image_size)
#         self.cmn = ops.CropMirrorNormalize(device='gpu',
#                                            dtype=types.FLOAT,
#                                            output_layout=types.NCHW,
#                                            mean=list(mean),
#                                            std=list(std))
#         self.lmdb_reader = lmdb_reader
#         self.image_size = image_size

#     def define_graph(self):
#         # IMPORTANT: call ExternalSource() here and keep returned DataNode handles
#         self._jpegs = self.input_jpegs()
#         self._labels = self.input_labels()
#         images = self.decode(self._jpegs)
#         images = self.resize(images)
#         images = self.cmn(images)
#         return images, self._labels

#     def iter_setup(self):
#         imgs, labels = self.lmdb_reader.next_batch(self.batch_size)
#         # Feed using the DataNode handles created in define_graph (self._jpegs/self._labels)
#         # This avoids the "Expected DataNode" TypeError in some DALI versions.
#         if self._jpegs is None or self._labels is None:
#             raise RuntimeError("DataNode handles not initialized. Make sure define_graph was called before iter_setup.")
#         self.feed_input(self._jpegs, imgs)
#         self.feed_input(self._labels, labels)

# import torch
# print("CUDA available:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("CUDA device count:", torch.cuda.device_count())
#     print("Current device:", torch.cuda.current_device())
#     print("Device name:", torch.cuda.get_device_name(0))

#     # 测试简单的GPU操作
#     x = torch.randn(3, 3).cuda()
#     print("GPU tensor:", x)

# import torch
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np

# # 标签索引到类别名称的映射
# label_mapping = {
#     0: 'airplane',
#     1: 'automobile',
#     2: 'bird',
#     3: 'cat',
#     4: 'deer',
#     5: 'dog',
#     6: 'frog',
#     7: 'horse',
#     8: 'ship',
#     9: 'truck'
# }

# transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# trainset = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transform)
# testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
# batch_size=4
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                          shuffle=True, num_workers=2)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                         shuffle=False, num_workers=2)

# # 遍历数据
# for images, labels in trainloader:
#     # images: [32, 3, 32, 32] 形状的张量
#     # labels: [32] 形状的张量，包含类别标签
#     print(images.shape, labels.shape)
#     break  # 只打印第一个批次的形状

# # 获取一些随机数据
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# # 打印标签
# print(' '.join('%5s' % label_mapping[labels[j].item()] for j in range(batch_size)))
# # 函数：显示图像
# def imshow(img):
#     img = img / 2 + 0.5  # 反归一化
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
# 显示图像
# imshow(torchvision.utils.make_grid(images))

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Conv2d(3, 6, 5)
#         self.pool = torch.nn.MaxPool2d(2, 2)
#         self.conv2 = torch.nn.Conv2d(6, 16, 5)
#         self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = torch.nn.Linear(120, 84)
#         self.fc3 = torch.nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# net = Net()

# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# for epoch in range(2):  # 多批次循环
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # 获取输入数据
#         inputs, labels = data

#         # 梯度清零
#         optimizer.zero_grad()

#         # 正向传播，反向传播，优化
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # 打印统计信息
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # 每2000个小批量打印一次
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
# print('Finished Training')

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print('Accuracy of the network on the 50000 train images: %d %%' % (100 * correct / total))


# class_correct = list(0. for _ in range(10))
# class_total = list(0. for _ in range(10))
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == labels).squeeze()
#         for i in range(batch_size):
#             label = labels[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1
# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         label_mapping[i], 100 * class_correct[i] / class_total[i]))
# torch.save(net.state_dict(), './checkpoint/cifar_net.pth')
# print('Model saved to ./checkpoint/cifar_net.pth')
# net2 = Net()
# net2.load_state_dict(torch.load('./checkpoint/cifar_net.pth'))
# print('Model loaded from ./checkpoint/cifar_net.pth')
