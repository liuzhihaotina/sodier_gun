import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.train_eval import train_model, train_model_auto, train_model_nn 
from modules.train_eval import eval_model, eval_model_nn
from modules.normalize import z_score_normalize, min_max_normalize_neg1_1, min_max_normalize_0_1, standardize_data

# ----------------------------二次多项式回归模型---------------------------------------
class QuadraticRegression_handgrad():
    """二次多项式回归模型--手写梯度版"""
    def __init__(self, params):
        self.params = params
    
    def forward(self, x):
        """推理值"""
        return self.params[0]*x**2 + self.params[1]*x + self.params[2] 
    
    def backward(self, x, y, y_pred, learning_rate): 
        """计算梯度、更新参数"""
        grad_a = 2 * ((y_pred - y)*x**2).mean()
        grad_b = 2 * ((y_pred - y)*x).mean()
        grad_c = 2 * (y_pred - y).mean()

        self.params[0] -= grad_a*learning_rate
        self.params[1] -= grad_b*learning_rate
        self.params[2] -= grad_c*learning_rate

    def return_params(self):
        return f'(a, b, c)=({self.params[0]}, {self.params[1]}, {self.params[2]})'
    
class QuadraticRegression_autograd():
    """二次多项式回归模型--自动更新梯度"""
    def __init__(self, params):
        self.params = params
    
    def forward(self, x):
        """推理值"""
        return self.params[0]*x**2 + self.params[1]*x + self.params[2] 

    def return_params(self):
        return f'(a, b, c)=({self.params[0]}, {self.params[1]}, {self.params[2]})'

# ------------------------训练集、测试集产生--------------------------------------
def create_quadratic_tensor(a, b, c, n_samples=16, x_range=(-5, 5)):
    """
    创建二次函数张量（生成训练集和测试集）

    Parameters:
    a, b, c: 二次函数系数
    n_samples: 样本数量
    x_range: 自变量取值范围
    """

    x_ = torch.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
    y_ = a * x_**2 + b * x_ + c

    return torch.cat([x_, y_], dim=1)

# ------------------------------主函数-----------------------------------
if __name__ == '__main__':
    # 迭代周期、学习率
    epochs = 50000
    learning_rate = 1e-2

    # 生成训练集和测试集
    a, b, c = 0.5, -2.0, -3.0
    samples = create_quadratic_tensor(a, b, c, n_samples=1000, x_range=(-100, 100))
    # from modules.draw_mean_std import mean_std_vis # 验证归一化
    # mean_std_vis(samples[:,1].numpy(), save_back='quadra')
    n_samples = samples.shape[0]
    n_val = int(0.2 * n_samples) # 20%测试集 80%训练集
    shuffled_indices = torch.randperm(n_samples)
    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:]
    train_set = samples[train_indices]
    test_set = samples[val_indices]
    # 归一化
    # train_set = min_max_normalize_neg1_1(train_set)
    # train_set = z_score_normalize(train_set)
    # train_set = standardize_data(train_set)
    # train_set = min_max_normalize_0_1(train_set)

    # 初始参数
    a, b, c = -1, 2, 3
    print(f"\n模型初始化参数:")
    print(f"二次系数: {a:.3f}")
    print(f"一次系数: {b:.3f}")
    print(f"常数: {c:.3f}")
    
    # 创建模型实例、并训练 抛物线回归效果排名：nn.ln > nn(ReLU) > auto > nn(Tanh) > hand
    choice = 'auto'
    if choice == 'hand':   # --手动计算梯度
        model = QuadraticRegression_handgrad([a, b, c])
        x = train_set[:, 0]
        y = train_set[:, 1]
        train_model(model, x, y, learning_rate, epochs)
        # 测试评估
        x = test_set[:, 0]
        y = test_set[:, 1]
        eval_model(model, x, y, draw=True)
    elif choice == 'auto': # --自动计算梯度
        params = torch.tensor([a, b, c], dtype=torch.float32, requires_grad=True)
        model = QuadraticRegression_autograd(params)
        x = train_set[:, 0]
        y = train_set[:, 1]
        train_model_auto(model, x, y, learning_rate, epochs)
        # 测试评估
        x = test_set[:, 0]
        y = test_set[:, 1]
        eval_model(model, x, y, draw=True)
    elif choice == 'nn.ln':
        linear_model = nn.Linear(2, 1)
        model = linear_model
        x = train_set[:, 0].unsqueeze(1)
        x = torch.cat([x**2, x], dim=1)
        y = train_set[:, 1].unsqueeze(1)
        train_model_nn(model, x, y, learning_rate, epochs)
        # 测试评估
        x = test_set[:, 0].unsqueeze(1)
        x = torch.cat([x**2, x], dim=1)
        y = test_set[:, 1].unsqueeze(1)
        eval_model_nn(model, x, y, draw=True)
    else:
        from collections import OrderedDict
        model = nn.Sequential(OrderedDict([
                        ('hidden_linear', nn.Linear(2, 1000)),
                        ('hidden_activation', nn.ReLU()),
                        ('output_linear', nn.Linear(1000, 1))
                        ]))
        x = train_set[:, 0].unsqueeze(1)
        x = torch.cat([x**2, x], dim=1)
        y = train_set[:, 1].unsqueeze(1)
        train_model_nn(model, x, y, learning_rate, epochs)
        # 测试评估
        x = test_set[:, 0].unsqueeze(1)
        x = torch.cat([x**2, x], dim=1)
        y = test_set[:, 1].unsqueeze(1)
        eval_model_nn(model, x, y, draw=True)


