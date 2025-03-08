import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
        self.sig = nn.Sigmoid()
        self.num_layers = num_layers



    def forward(self, x):
        out = self.embedding(x)
        out, hidden = self.lstm(out)
        out = self.dropout(out[:, -1, :])
        #out = self.dropout(out[-1])
        out = self.output(out)
        #out = self.sig(out)
        return out
