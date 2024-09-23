import torch
import torch.nn as nn

class StockPricePredictor1(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(StockPricePredictor1, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size,batch_first=True)              # LSTM层 输入维度为input_size，输出维度为hidden_layer_size
        self.linear = nn.Linear(hidden_layer_size, output_size)         # 线性层 输入维度为hidden_layer_size，输出维度为output_size
        self.hidden_cell = (torch.zeros(1,1,hidden_layer_size),         # hidden_cell是一个元组，包含两个张量
                            torch.zeros(1,1,hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell) # .view()表示将张量转换为指定形状的张量 len(input_seq)表示行数 1表示列数 -1表示自动计算列数
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]      # 返回最后一个预测值

    # input_seq.view(len(input_seq) ,1, -1)的作用是将input_seq转换为三维张量，第一维度是序列长度，第二维度是batch_size，第三维度是特征数
    # 要改成 第一维度是batch_size，第二维度是序列长度，第三维度是特征数，只需要将input_seq.view(len(input_seq) ,1, -1)改成input_seq.view(1, len(input_seq), -1)即可
class StockPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(StockPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # 隐藏状态初始化
        batch_size = input_seq.size(0)
        h0 = torch.zeros(1, batch_size, 50).to(input_seq.device)
        c0 = torch.zeros(1, batch_size, 50).to(input_seq.device)

        lstm_out, (hn, cn) = self.lstm(input_seq, (h0, c0))
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions