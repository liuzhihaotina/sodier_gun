import torch

# -----------------------------------------归一化方法--------------------------------------------
def z_score_normalize(tensor):
    """均值0标准差1, 推荐学习率1e-1"""
    return (tensor - tensor.mean()) / tensor.std()

def min_max_normalize_neg1_1(tensor):
    """归一化到 [-1, 1] 范围, 推荐学习率1e-3"""
    min_val = tensor.min()
    max_val = tensor.max()
    return 2 * (tensor - min_val) / (max_val - min_val) - 1

def min_max_normalize_0_1(tensor):
    """归一化到 [0, 1] 范围, 推荐学习率1e-2"""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

def standardize_data(X):
    """
    标准化: (x - mean) / std
    结果: 均值=0, 标准差=1
    """
    mean = X.mean(dim=0)
    std = X.std(dim=0)
    # 避免除零
    std = torch.where(std == 0, torch.ones_like(std), std)
    X_normalized = (X - mean) / std
    return X_normalized