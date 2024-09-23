import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable
from LSTM import StockPricePredictor
from torch.utils.tensorboard import SummaryWriter
accuracy = lambda y, y_pred: 1 - torch.abs((y - y_pred) / y).mean()

# 数据加载与预处理
df = pd.read_csv(r'C:\Users\Zayn\Downloads\example\simulated_stock_data.csv')
data = df['Close'].values
scaler = MinMaxScaler(feature_range=(-1, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1))

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60
X, y = create_sequences(data_normalized, seq_length)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据拆分
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train = torch.from_numpy(X_train).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

# LSTM模型定义
model = StockPricePredictor(input_size=1, hidden_layer_size=50, output_size=1).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模型训练
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    X_train = X_train
    y_train = y_train
    y_pred = model(X_train)
    single_loss = loss_function(y_pred, y_train)
    single_loss.backward()
    optimizer.step()

    # 输出训练信息
    acc = accuracy(y_train, y_pred)
    if epoch % 10 == 0:
        print(f'epoch: {epoch:3} loss: {single_loss.item():10.8f}')
    SummaryWriter('runs/stock_sell').add_scalar('Loss/train', single_loss, epoch)       # 记录训练损失 runs/stock_sell 文件夹下
    SummaryWriter('runs/stock_sell').add_scalar('Accuracy/train', acc, epoch)

# 预测与可视化
model.eval()
with torch.no_grad():
    train_predictions = model(X_train).detach().cpu().numpy()
    test_predictions = model(X_test).detach().cpu().numpy()
    
    # 还原预测值和实际值
    data_inv = scaler.inverse_transform(data_normalized)
    train_predictions_inv = scaler.inverse_transform(np.concatenate([np.zeros((len(train_predictions), 1)), train_predictions], axis=1))[:, 1]
    test_predictions_inv = scaler.inverse_transform(np.concatenate([np.zeros((len(test_predictions), 1)), test_predictions], axis=1))[:, 1]
    
    plt.figure(figsize=(12,6))
    plt.plot(data_inv, label='Actual Data')
    plt.plot(np.arange(seq_length, len(train_predictions_inv) + seq_length), train_predictions_inv, label='Train Predictions')
    plt.plot(np.arange(len(data_inv) - len(test_predictions_inv), len(data_inv)), test_predictions_inv, label='Test Predictions')
    plt.legend()
    plt.show()
print('Done!')