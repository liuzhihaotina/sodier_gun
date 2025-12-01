import torch
import matplotlib.pyplot as plt

# ----------------------------------评估结果可视化--------------------------------------
def set_chinese():
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from modules.chinese_font import chinese_font
    plt.rcParams['font.family'] = [chinese_font.get_name(), 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

def plot_regression_line(y_true, y_pred, x_data):
    """
    绘制回归线和数据点

    Parameters:
    y_true: 真实值
    y_pred: 预测值  
    x_data: 输入特征
    """
    plt.figure(figsize=(12, 5))

    # 按x排序以便绘制平滑的回归线
    sorted_indices = torch.argsort(x_data.squeeze())
    x_sorted = x_data[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]
    y_true_sorted = y_true[sorted_indices]

    # 子图1：回归线对比
    plt.subplot(1, 2, 1)
    plt.scatter(x_data.numpy(), y_true.numpy(),
                alpha=0.7, color='blue', label='真实值', s=30)
    plt.scatter(x_data.numpy(), y_pred.detach().numpy(),
                alpha=1, color='red', label='预测值', s=150, marker='x')
    plt.plot(x_sorted.numpy(), y_pred_sorted.detach().numpy(),
             'r--', linewidth=2, alpha=0.5, label='回归线')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('线性回归拟合结果')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 子图2：残差图
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred.detach().numpy(), residuals.detach().numpy(),
                alpha=0.7, color='green', label='gt-pred', s=40)
    plt.axhline(y=0, color='black', linestyle='--', label='水平参考线', alpha=0.8)
    plt.xlabel('预测值')
    plt.ylabel('残差 (真实值 - 预测值)')
    plt.title('残差图')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def result_plot(x, y_true, y_pred, draw=False):
    """输出模型评估指标、绘制预测结果"""
    # 打印统计信息
    mse = torch.mean((y_true - y_pred) ** 2).item()
    r2 = 1 - torch.sum((y_true - y_pred) ** 2) / torch.sum((y_true - y_true.mean()) ** 2)
    print(f"模型评估指标:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"R² 分数: {r2.item():.4f}")
    
    if draw:
        set_chinese()
        if x.shape!=y_pred.shape:
            x = x[:, -1]
        plot_regression_line(y_true, y_pred, x)