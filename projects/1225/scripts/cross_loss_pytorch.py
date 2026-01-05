import torch

logits = torch.tensor([
    [2.0, 1.0, 0.1],
    [0.5, 2.5, -1.0],
    [-1.2, 0.3, 1.7],
])

labels = torch.tensor([0, 1, 2])

loss_none = torch.nn.CrossEntropyLoss(reduction="none")(logits, labels)
loss_mean = torch.nn.CrossEntropyLoss(reduction="mean")(logits, labels)

print("per-sample:", loss_none)
print("mean:", loss_mean)

