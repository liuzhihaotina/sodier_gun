import torch
from PIL import Image
from torchvision import models
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])
img = Image.open("data/test/dogs/dog.4001.jpg")
# img.show()
img_t = preprocess(img)

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

batch_t = torch.unsqueeze(img_t, 0)

# print(dir(models))
# resnet = models.resnet101(pretrained=True)
with open('data/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
# print(resnet)
resnet.eval()
out = resnet(batch_t)
# print(out.shape)
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
a, b = labels[index[0]], percentage[index[0]].item()
print(f'Predicted: {a}, {b}%')

_, indices = torch.sort(out, descending=True)
c = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
for i,j in c:
    print(i, j)


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
