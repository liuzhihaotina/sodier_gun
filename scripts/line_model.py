import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.train_eval import train_model, eval_model
from modules.vis_eval import result_plot

# ----------------------------线性回归模型---------------------------------------
class LinearRegression():
    """线性回归模型"""
    def __init__(self, w, b):
        self.w = w
        self.b = b
    
    def forward(self, x):
        """推理值"""
        return self.w*x+self.b
    
    def backward(self, x, y, y_pred, learning_rate): 
        """计算梯度、更新参数"""
        grad_w = 2 * ((y_pred - y)*x).mean()
        grad_b = 2 * (y_pred - y).mean()

        self.w -= grad_w*learning_rate
        self.b -= grad_b*learning_rate

    def return_params(self):
        return f'(w, b)=({self.w}, {self.b})'

# ------------------------训练集、测试集产生--------------------------------------
def create_line_tensor(w, b, n_samples=16, x_range=(-5, 5)):
    """
    创建一次函数张量（生成训练集和测试集）

    Parameters:
    w, b: 一次函数系数
    n_samples: 样本数量
    x_range: 自变量取值范围
    """

    x_ = torch.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
    y_ =  w * x_ + b

    return torch.cat([x_, y_], dim=1)

# ------------------------------主函数-----------------------------------
if __name__ == '__main__':
    # 迭代周期、学习率
    epochs = 1000
    learning_rate = 0.015

    # 创建模型实例
    w, b = 1, 0
    model = LinearRegression(w, b)
    print(f"\n模型初始化参数:")
    print(f"权重: {model.w:.3f}")
    print(f"偏置: {model.b:.3f}")

    # 训练模型
    train_set = create_line_tensor(w, b, n_samples=100, x_range=(-50, 50))
    train_model(model, train_set, learning_rate, epochs)

    # 测试评估
    test_set = create_line_tensor(w, b, n_samples=50, x_range=(51, 100))
    eval_model(model, test_set)
    
    # 输出评估指标、可视化
    result_plot(test_set, model, draw=True)