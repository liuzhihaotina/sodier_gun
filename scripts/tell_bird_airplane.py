import torch
from torchvision import datasets
from torchvision import transforms 
import matplotlib.pyplot as plt
import torch.nn as nn
data_path = 'data_cifar-10'
# class_name={0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 
#             8:'ship', 9:'truck'}
img_mean = (0.4915, 0.4823, 0.4468)
img_std = (0.2470, 0.2435, 0.2616)
img_mean_val = (0.6310, 0.6126, 0.6709)
img_std_val = (0.2546, 0.2378, 0.2616)
train_set = datasets.CIFAR10(data_path, train=True, download=True, 
            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(img_mean, img_std)]))
test_set = datasets.CIFAR10(data_path, train=False, download=True, 
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(img_mean_val, img_std_val)]))

label_map = {0: 0, 2: 1}
class_names = ['airplane', 'bird']
train = [(img, label_map[label]) for img, label in train_set if label in [0, 2]]
test = [(img, label_map[label]) for img, label in test_set if label in [0, 2]]
batch_size = 4
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True) # 乱序
val_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False) # 顺序

n_out = 2
# model = nn.Sequential(nn.Linear(3072, 512,), nn.Tanh(), nn.Linear(512, n_out), nn.LogSoftmax(dim=1))
model = nn.Sequential(
nn.Linear(3072, 1024),
nn.ReLU(),
nn.Linear(1024, 512),
nn.ReLU(),
nn.Linear(512, 128),
nn.ReLU(),
nn.Linear(128, 2))#,
# nn.LogSoftmax(dim=1))
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# loss_fn = nn.NLLLoss()
loss_fn = nn.CrossEntropyLoss()
n_epochs = 10
for epoch in range(n_epochs):
    for img, label in train_loader:
        out = model(img.view(batch_size, -1))
        loss = loss_fn(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Epoch: %d, Loss: %f" % (epoch, float(loss)))

correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        outputs = model(imgs.view(batch_size, -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
print("Accuracy: %f", correct / total)