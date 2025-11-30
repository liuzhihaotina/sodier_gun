# -------------------------------训练模型、评估模型---------------------------------------------
def loss_cal(y, y_pred):
    return (y_pred - y)**2

def train_model(model, input, learning_rate, epochs):
    """训练模型"""
    x = input[:, 0]
    y = input[:, 1]
    y_pred = model.forward(x) 
    loss = (model.forward(x)-y)**2
    print(f'init--> {model.return_params()}, total_loss: {loss.sum()}, mean_loss: {loss.mean()}') 
    for epoch in range(1, epochs+1):

        y_pred = model.forward(x) # 前向传播，返回推理值
        model.backward(x, y, y_pred, learning_rate) # 后向传播计算梯度值、结合学习率更新权重参数

        loss = loss_cal(y, y_pred) # 计算损失值
        if epoch % (epochs/10) ==0:
            print(f'epoch: {epoch}, {model.return_params()}, total_loss: {loss.sum()}, mean_loss: {loss.mean()}')

def eval_model(model, test_set):
    x = test_set[:, 0]
    y = test_set[:, 1]

    y_pred = model.forward(x)

    loss = loss_cal(y, y_pred)
    print(f'{model.return_params()}, total_loss: {loss.sum()}, mean_loss: {loss.mean()}')