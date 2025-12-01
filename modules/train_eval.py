import torch
import torch.optim as optim
import torch.nn as nn
from modules.vis_eval import result_plot
import math

# -------------------------------训练模型、评估模型---------------------------------------------
loss_calnn = nn.MSELoss()

def loss_cal(y, y_pred):
    return (y_pred - y)**2

def train_model(model, x, y, learning_rate, epochs):
    """训练模型"""
    y_pred = model.forward(x) 
    loss = loss_cal(y, y_pred)
    print(f'init--> {model.return_params()}, total_loss: {loss.sum()}, mean_loss: {loss.mean()}') 
    max_gradnorm = 1
    for epoch in range(1, epochs+1):

        y_pred = model.forward(x) # 前向传播，返回推理值
        model.backward(x, y, y_pred, learning_rate) # 后向传播计算梯度值、结合学习率更新权重参数
        # 防止梯度爆炸
        for i, p in enumerate(model.params):
            model.params[i] = math.copysign(max_gradnorm, p)*min(max_gradnorm, abs(p))
        
        loss = loss_cal(y, y_pred) # 计算损失值
        if epoch % (epochs/10) == 0:
            print(f'epoch: {epoch}, {model.return_params()}, total_loss: {loss.sum()}, mean_loss: {loss.mean()}')

def train_model_auto(model, x, y, learning_rate, epochs):
    """训练模型"""
    y_pred = model.forward(x) 
    loss = loss_cal(y, y_pred)
    print(f'init--> {model.return_params()}, total_loss: {loss.sum()}, mean_loss: {loss.mean()}') 
    params=model.params
    optimizer = optim.Adam([params], lr=learning_rate)
    for epoch in range(1, epochs+1):
        # if model.params.grad is not None:
        #     model.params.grad.zero_()

        y_pred = model.forward(x) # 前向传播，返回推理值
        model.params=params

        loss = loss_cal(y, y_pred) # 计算损失值
        optimizer.zero_grad()
        loss.mean().backward() # 后向传播计算梯度值
        torch.nn.utils.clip_grad_norm_([model.params], max_norm=1) # 防止梯度爆炸
        optimizer.step()

        # with torch.no_grad(): # 结合学习率更新权重参数
        #     model.params -= model.params.grad*learning_rate

        if epoch % (epochs/10) == 0:
            print(f'epoch: {epoch}, {model.return_params()}, total_loss: {loss.sum()}, mean_loss: {loss.mean()}')

def train_model_nn(model, x, y, learning_rate, epochs):
    """训练模型"""
    y_pred = model(x) 
    loss = loss_calnn(y, y_pred)
    print(f'init--> (w, b)={model.weight.data.squeeze().tolist(), model.bias.item()}, total_loss: {loss.sum()}, mean_loss: {loss.mean()}') 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(1, epochs+1):

        y_pred = model(x) # 前向传播，返回推理值

        loss = loss_calnn(y, y_pred) # 计算损失值
        optimizer.zero_grad()
        loss.backward() # 后向传播计算梯度值
        # torch.nn.utils.clip_grad_norm_([model.params], max_norm=1) # 防止梯度爆炸
        optimizer.step()

        if epoch % (epochs/10) == 0:
            print(f'epoch: {epoch}, (w, b)={model.weight.data.squeeze().tolist(), model.bias.item()}, total_loss: {loss.sum()}, mean_loss: {loss.mean()}')

def eval_model(model, x, y, draw):
    y_pred = model.forward(x)

    loss = loss_cal(y, y_pred)
    print(f'{model.return_params()}, total_loss: {loss.sum()}, mean_loss: {loss.mean()}')

    # 输出评估指标、可视化
    result_plot(x, y, y_pred, draw=draw)

def eval_model_nn(model, x, y, draw):
    model.eval()
    y_pred = model(x)

    loss = loss_calnn(y, y_pred)
    print(f'(w, b)={model.weight.data.squeeze().tolist(), model.bias.item()}, total_loss: {loss.sum()}, mean_loss: {loss.mean()}')

    # 输出评估指标、可视化
    result_plot(x, y, y_pred, draw=draw)