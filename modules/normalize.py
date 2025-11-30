# -----------------------------------------归一化方法--------------------------------------------
def z_score_normalize(tensor):
    return (tensor - tensor.mean()) / tensor.std()

def min_max_normalize_neg1_1(tensor):
    """直接归一化到 [-1, 1] 范围"""
    min_val = tensor.min()
    max_val = tensor.max()
    return 2 * (tensor - min_val) / (max_val - min_val) - 1

def min_max_normalize_0_1(tensor):
    """直接归一化到 [0, 1] 范围"""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)