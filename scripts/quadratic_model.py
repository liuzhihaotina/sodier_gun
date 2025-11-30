import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.train_eval import train_model, eval_model
from modules.vis_eval import result_plot

# ----------------------------二次多项式回归模型---------------------------------------
class QuadraticRegression():
    """二次多项式回归模型"""
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
    
    def forward(self, x):
        """推理值"""
        return self.a*x**2 + self.b*x + c 
    
    def backward(self, x, y, y_pred, learning_rate): 
        """计算梯度、更新参数"""
        grad_a = 2 * ((y_pred - y)*x**2).mean()
        grad_b = 2 * ((y_pred - y)*x).mean()
        grad_c = 2 * (y_pred - y).mean()

        self.a -= grad_a*learning_rate
        self.b -= grad_b*learning_rate
        self.c -= grad_c*learning_rate

    def return_params(self):
        return f'(a, b, c)=({self.a}, {self.b}, {self.c})'

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
    epochs = 1000
    learning_rate = 0.015

    # 创建模型实例
    a, b, c = 0.5, -2.0, -3.0
    model = QuadraticRegression(a, b, c)
    print(f"\n模型初始化参数:")
    print(f"二次系数: {model.a:.3f}")
    print(f"一次系数: {model.b:.3f}")
    print(f"常数: {model.c:.3f}")

    # 训练模型
    train_set = create_quadratic_tensor(a, b, c, n_samples=100, x_range=(51, 100))
    train_model(model, train_set, learning_rate, epochs)

    # 测试评估
    test_set = create_quadratic_tensor(a, b, c, n_samples=50, x_range=(-50, 50))
    eval_model(model, test_set)
    
    # 输出评估指标、可视化
    result_plot(test_set, model, draw=True)
